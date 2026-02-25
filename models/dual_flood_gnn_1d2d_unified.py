import torch
from torch import Tensor
from torch import nn
from torch.nn import LayerNorm
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from typing import Tuple
from utils.model_utils import make_mlp

from .base_model_1d2d import BaseModel1D2D


class UnifiedDUALFloodGNN1D2D(BaseModel1D2D):
    """
    Unified shared encoder-decoder GNN for 1D and 2D flood prediction.

    Key design:
    - Single shared node encoder (1D and 2D features are pre-aligned)
    - Single shared edge encoder (1D and 2D edge features are pre-aligned)
    - Node type embedding to preserve physical context (1D channel vs 2D surface)
    - Single shared GNN operating over a merged graph (2D nodes + 1D nodes + coupling edges)
    - Separate lightweight decoder heads for 2D and 1D output predictions
    """

    def __init__(
        self,
        # Node/edge feature sizes (shared between 1D and 2D after preprocessing)
        input_align_features: int = None,
        input_align_edge_features: int = None,
        # Output sizes (may differ between 1D and 2D)
        output_features: int = None,
        output_1d_features: int = None,
        # Architecture parameters
        hidden_features: int = None,
        num_layers: int = 2,
        activation: str = "relu",
        residual: bool = True,
        mlp_layers: int = 2,
        # Encoder / Decoder parameters
        encoder_layers: int = 0,
        encoder_activation: str = None,
        decoder_layers: int = 0,
        decoder_activation: str = None,
        # Misc
        use_layer_norm: bool = True,
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)

        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.use_layer_norm = use_layer_norm

        # ── Resolve defaults from base model ──────────────────────────────────
        if input_align_features is None:
            input_align_features = self.input_node_features
        if input_align_edge_features is None:
            input_align_edge_features = self.input_edge_features
        if output_features is None:
            output_features = self.output_node_features
        if output_1d_features is None:
            output_1d_features = self.output_1d_node_features

        encoder_decoder_hidden = hidden_features * 2

        # ── Shared Node Encoder ───────────────────────────────────────────────
        # Both 1D and 2D nodes share identical input feature schema after preprocessing, so a single encoder is sufficient.
        if self.with_encoder:
            self.node_encoder = make_mlp(
                input_size=input_align_features,
                output_size=hidden_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=encoder_layers,
                activation=encoder_activation,
                bias=False,
                device=self.device,
            )

        # ── Shared Edge Encoder ───────────────────────────────────────────────
        # Covers 2D edges, 1D edges, and coupling edges uniformly.
        if self.with_encoder:
            self.edge_encoder = make_mlp(
                input_size=input_align_edge_features,
                output_size=hidden_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=encoder_layers,
                activation=encoder_activation,
                bias=False,
                device=self.device,
            )

        # ── Node Type Embedding ───────────────────────────────────────────────
        # Even though 1D and 2D share the same feature schema, they represent physically different contexts.
        # The type embedding lets the GNN condition on this distinction.
        #   0 → 2D surface node
        #   1 → 1D channel/pipe node
        self.type_embed = nn.Embedding(2, hidden_features, device=self.device)

        # ── Determine GNN input/output sizes ─────────────────────────────────
        # If no encoder, the GNN receives raw features directly.
        # Output size stays at hidden_features when a decoder follows.
        input_node_size = hidden_features if self.with_encoder else input_align_features
        input_edge_size = hidden_features if self.with_encoder else input_align_edge_features
        output_node_size = hidden_features if self.with_decoder else max(output_features, output_1d_features)

        # ── Shared GNN ────────────────────────────────────────────────────────
        self.convs = self._make_gnn(
            input_node_size=input_node_size,
            input_edge_size=input_edge_size,
            output_node_size=output_node_size,
            hidden_features=hidden_features,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            activation=activation,
            residual=residual,
            device=self.device,
        )

        # ── Layer Normalisation ───────────────────────────────────────────────
        if self.use_layer_norm:
            self.norm = LayerNorm(hidden_features, device=self.device)

        # ── Separate Decoder Heads ────────────────────────────────────────────
        # Separate heads allow different output dimensionalities and let each
        # domain fine-tune its own output mapping from the shared latent space.
        if self.with_decoder:
            self.node_decoder_2d = make_mlp(
                input_size=hidden_features,
                output_size=output_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=decoder_layers,
                activation=decoder_activation,
                bias=False,
                device=self.device,
            )
            self.node_decoder_1d = make_mlp(
                input_size=hidden_features,
                output_size=output_1d_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=decoder_layers,
                activation=decoder_activation,
                bias=False,
                device=self.device,
            )

    # ── GNN builder ────────────────────────────
    def _make_gnn(
        self,
        input_node_size: int,
        input_edge_size: int,
        output_node_size: int,
        hidden_features: int,
        num_layers: int,
        mlp_layers: int,
        activation: str,
        residual: bool,
        device: str,
    ):
        if num_layers == 1:
            return NodeEdgeConv(
                node_in_channels=input_node_size,
                edge_in_channels=input_edge_size,
                node_out_channels=output_node_size,
                hidden_size=hidden_features,
                num_layers=mlp_layers,
                activation=activation,
                residual=residual,
                bias=False,
                device=device,
            )

        layers = []
        for _ in range(num_layers):
            layers.append(
                (
                    NodeEdgeConv(
                        node_in_channels=input_node_size,
                        edge_in_channels=input_edge_size,
                        node_out_channels=output_node_size,
                        hidden_size=hidden_features,
                        num_layers=mlp_layers,
                        activation=activation,
                        residual=residual,
                        bias=False,
                        device=device,
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
        return PygSequential("x, edge_index, edge_attr", layers)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        x_1d: Tensor,
        edge_index_1d: Tensor,
        edge_attr_1d: Tensor,
        edge_index_1d_2d: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for unified 1D-2D water level prediction.

        Args:
            x               : 2D node features          [N_2d, input_features]
            edge_index      : 2D edge connectivity       [2, E_2d]
            edge_attr       : 2D edge features           [E_2d, input_edge_features]
            x_1d            : 1D node features           [N_1d, input_features]
            edge_index_1d   : 1D edge connectivity       [2, E_1d]
            edge_attr_1d    : 1D edge features           [E_1d, input_edge_features]
            edge_index_1d_2d: Coupling edge connectivity [2, E_c]
                              Row 0 = 1D node indices, Row 1 = 2D node indices

        Returns:
            pred_2d : Predicted values for 2D nodes  [N_2d, output_features]
            pred_1d : Predicted values for 1D nodes  [N_1d, output_1d_features]
        """
        n_2d = x.size(0)
        n_1d = x_1d.size(0)

        # ── 1. Shared Encoding ────────────────────────────────────────────────
        if self.with_encoder:
            x       = self.node_encoder(x)
            x_1d    = self.node_encoder(x_1d)
            edge_attr     = self.edge_encoder(edge_attr)
            edge_attr_1d  = self.edge_encoder(edge_attr_1d)
            # edge_attr_1d_2d = self.edge_encoder(edge_attr_1d_2d)

        # ── 2. Merge nodes into a single graph ────────────────────────────────
        # Node order: [2D nodes (0 … N_2d-1) | 1D nodes (N_2d … N_2d+N_1d-1)]
        x_combined = torch.cat([x, x_1d], dim=0)       # [N_2d + N_1d, H]

        # ── 3. Add type embeddings ────────────────────────────────────────────
        type_ids = torch.cat([
            torch.zeros(n_2d, dtype=torch.long, device=x.device),   # 0 → 2D
            torch.ones(n_1d,  dtype=torch.long, device=x.device),   # 1 → 1D
        ])                                                            # [N_2d + N_1d]
        x_combined = x_combined + self.type_embed(type_ids)

        # ── 4. Merge edges into a single graph ────────────────────────────────
        # Offset 1D node indices so they sit after the 2D nodes.
        edge_index_1d_offset = edge_index_1d + n_2d

        # # Coupling edges: 1D side (row 0) needs offset; 2D side (row 1) does not.
        # coupling_src = edge_index_1d_2d[0] + n_2d   # 1D nodes (offset)
        # coupling_dst = edge_index_1d_2d[1]           # 2D nodes (no offset)

        # # Make coupling bidirectional so both domains exchange information.
        # coupling_fwd = torch.stack([coupling_src, coupling_dst])   # 1D → 2D
        # coupling_bwd = torch.stack([coupling_dst, coupling_src])   # 2D → 1D

        edge_index_merged = torch.cat([
            edge_index,            # 2D internal edges
            edge_index_1d_offset,  # 1D internal edges (offset)
        ], dim=1)                  # [2, E_2d + E_1d]

        # Edge features: coupling edges reuse the same features in both directions.
        edge_attr_merged = torch.cat([
            edge_attr,             # 2D
            edge_attr_1d,          # 1D
        ], dim=0)                  # [E_2d + E_1d, H]

        # ── 5. Shared GNN message passing ─────────────────────────────────────
        x_out = self.convs(x_combined, edge_index_merged, edge_attr_merged)

        # ── 6. Layer normalisation ────────────────────────────────────────────
        if self.use_layer_norm:
            x_out = self.norm(x_out)

        # ── 7. Split back into 2D and 1D node sets ────────────────────────────
        x_out_2d = x_out[:n_2d]    # [N_2d, H]
        x_out_1d = x_out[n_2d:]    # [N_1d, H]

        # ── 8. Separate decoder heads ─────────────────────────────────────────
        if self.with_decoder:
            pred_2d = self.node_decoder_2d(x_out_2d)
            pred_1d = self.node_decoder_1d(x_out_1d)
        else:
            pred_2d = x_out_2d
            pred_1d = x_out_1d

        return pred_2d, pred_1d


# ── NodeEdgeConv (unchanged from original) ────────────────────────────────────

class NodeEdgeConv(MessagePassing):
    """
    Message  = MLP(node_i || edge_attr || node_j)
    Aggregate = sum
    Update   = MLP(aggregated_message) [+ residual]
    """

    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        node_out_channels: int,
        hidden_size: int,
        num_layers: int = 2,
        activation: str = "relu",
        residual: bool = True,
        edge_residual_scale: float = 0.2,
        bias: bool = False,
        device: str = "cpu",
    ):
        super().__init__(aggr="sum")
        self.residual = residual
        self.edge_residual_scale = edge_residual_scale

        self.msg_mlp = make_mlp(
            input_size=2 * node_in_channels + edge_in_channels,
            output_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            device=device,
        )

        self.node_mlp = make_mlp(
            input_size=hidden_size,
            output_size=node_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            device=device,
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def propagate(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        msg = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        aggr = self.aggregate(msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data("update", coll_dict)
        return self.update(aggr, **update_kwargs)

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:
        cat_feats = torch.cat([x_i, edge_attr, x_j], dim=-1)
        msg = self.msg_mlp(cat_feats)
        if self.residual:
            msg = msg + self.edge_residual_scale * edge_attr
        return msg

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        out = self.node_mlp(aggr)
        if self.residual:
            out = out + x
        return out