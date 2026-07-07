import torch
from torch import Tensor
from torch.nn import LayerNorm
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from typing import Tuple
from utils.model_utils import make_mlp
from torch_scatter import scatter_mean

from .base_model_1d2d import BaseModel1D2D


class DUALFloodGNNNodeEdge1D2D(BaseModel1D2D):
    """
    Dual 1D/2D flood GNN that uses node AND edge features and predicts BOTH
    node-level targets (water level) and link-level targets (discharge) for the
    1D and 2D graphs. Four variables are predicted in total:
        - 2D node water-level delta
        - 2D edge discharge delta
        - 1D node water-level delta
        - 1D edge discharge delta

    1D <-> 2D information exchange happens through gated node coupling.
    """

    def __init__(
        self,
        # 2D parameters
        input_features: int = None,
        input_edge_features: int = None,
        output_features: int = None,
        output_edge_features: int = None,
        # 1D parameters
        input_1d_features: int = None,
        input_1d_edge_features: int = None,
        output_1d_features: int = None,
        output_1d_edge_features: int = None,
        # Shared parameters
        hidden_features: int = None,
        num_layers: int = 2,
        activation: str = "relu",
        residual: bool = True,
        mlp_layers: int = 2,
        # Encoder Decoder Parameters
        encoder_layers: int = 0,
        encoder_activation: str = None,
        decoder_layers: int = 0,
        decoder_activation: str = None,
        # Coupling parameters
        coupling_layers: int = 1,
        coupling_hidden: int = None,
        use_coupling_gate: bool = False,
        use_layer_norm: bool = True,
        # Edge stability parameters
        edge_residual_scale: float = 0.2,
        clip_edge_predictions: bool = False,
        max_edge_value: float = None,
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.with_coupling = coupling_layers > 0
        self.use_coupling_gate = use_coupling_gate
        self.use_layer_norm = use_layer_norm
        self.edge_residual_scale = edge_residual_scale
        self.clip_edge_predictions = clip_edge_predictions
        self.max_edge_value = max_edge_value

        # Set default values from base model
        if input_features is None:
            input_features = self.input_node_features
        if input_edge_features is None:
            input_edge_features = self.input_edge_features
        if output_features is None:
            output_features = self.output_node_features
        if output_edge_features is None:
            output_edge_features = self.output_edge_features

        if input_1d_features is None:
            input_1d_features = self.input_1d_node_features
        if input_1d_edge_features is None:
            input_1d_edge_features = self.input_1d_edge_features
        if output_1d_features is None:
            output_1d_features = self.output_1d_node_features
        if output_1d_edge_features is None:
            output_1d_edge_features = self.output_1d_edge_features

        if coupling_hidden is None:
            coupling_hidden = hidden_features

        encoder_decoder_hidden = hidden_features * 2

        # ========== 2D Encoders ==========
        if self.with_encoder:
            self.node_encoder_2d = make_mlp(
                input_size=input_features,
                output_size=hidden_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=encoder_layers,
                activation=encoder_activation,
                bias=False,
                device=self.device,
            )

            self.edge_encoder_2d = make_mlp(
                input_size=input_edge_features,
                output_size=hidden_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=encoder_layers,
                activation=encoder_activation,
                bias=False,
                device=self.device,
            )

        # ========== 1D Encoders ==========
        if self.with_encoder:
            self.node_encoder_1d = make_mlp(
                input_size=input_1d_features,
                output_size=hidden_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=encoder_layers,
                activation=encoder_activation,
                bias=False,
                device=self.device,
            )

            self.edge_encoder_1d = make_mlp(
                input_size=input_1d_edge_features,
                output_size=hidden_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=encoder_layers,
                activation=encoder_activation,
                bias=False,
                device=self.device,
            )

        # Determine input/output sizes for GNN layers
        input_node_size_2d = hidden_features if self.with_encoder else input_features
        input_edge_size_2d = (
            hidden_features if self.with_encoder else input_edge_features
        )
        output_node_size_2d = hidden_features if self.with_decoder else output_features
        output_edge_size_2d = (
            hidden_features if self.with_decoder else output_edge_features
        )

        input_node_size_1d = hidden_features if self.with_encoder else input_1d_features
        input_edge_size_1d = (
            hidden_features if self.with_encoder else input_1d_edge_features
        )
        output_node_size_1d = (
            hidden_features if self.with_decoder else output_1d_features
        )
        output_edge_size_1d = (
            hidden_features if self.with_decoder else output_1d_edge_features
        )

        # ========== 2D GNN Layers ==========
        self.convs_2d = self._make_gnn(
            input_node_size=input_node_size_2d,
            input_edge_size=input_edge_size_2d,
            output_node_size=output_node_size_2d,
            output_edge_size=output_edge_size_2d,
            hidden_features=hidden_features,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            activation=activation,
            residual=residual,
            device=self.device,
        )

        # ========== 1D GNN Layers ==========
        self.convs_1d = self._make_gnn(
            input_node_size=input_node_size_1d,
            input_edge_size=input_edge_size_1d,
            output_node_size=output_node_size_1d,
            output_edge_size=output_edge_size_1d,
            hidden_features=hidden_features,
            num_layers=num_layers,
            mlp_layers=mlp_layers,
            activation=activation,
            residual=residual,
            device=self.device,
        )

        # ========== Layer Normalization ==========
        if self.use_layer_norm:
            self.norm_2d = LayerNorm(hidden_features, device=self.device)
            self.norm_1d = LayerNorm(hidden_features, device=self.device)
            self.norm_2d_post_coupling = LayerNorm(hidden_features, device=self.device)
            self.norm_1d_post_coupling = LayerNorm(hidden_features, device=self.device)

        # ========== Coupling Layers (1D -> 2D and 2D -> 1D) ==========
        if self.with_coupling:
            # Maps 1D features to 2D nodes (via edge_index_1d_2d)
            self.coupling_1d_to_2d = make_mlp(
                input_size=hidden_features,
                output_size=hidden_features,
                hidden_size=coupling_hidden,
                num_layers=coupling_layers,
                activation=activation,
                bias=False,
                device=self.device,
            )
            # Maps 2D features to 1D nodes (via edge_index_1d_2d)
            self.coupling_2d_to_1d = make_mlp(
                input_size=hidden_features,
                output_size=hidden_features,
                hidden_size=coupling_hidden,
                num_layers=coupling_layers,
                activation=activation,
                bias=False,
                device=self.device,
            )

            # Gating mechanism
            if self.use_coupling_gate:
                # Gate for 1D->2D coupling
                self.gate_1d_to_2d = make_mlp(
                    input_size=hidden_features * 2,
                    output_size=hidden_features,
                    hidden_size=coupling_hidden,
                    num_layers=1,
                    activation="sigmoid",
                    bias=False,
                    device=self.device,
                )
                # Gate for 2D->1D coupling
                self.gate_2d_to_1d = make_mlp(
                    input_size=hidden_features * 2,
                    output_size=hidden_features,
                    hidden_size=coupling_hidden,
                    num_layers=1,
                    activation="sigmoid",
                    bias=False,
                    device=self.device,
                )

        # ========== 2D Decoders (node + edge) ==========
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
            self.edge_decoder_2d = make_mlp(
                input_size=hidden_features,
                output_size=output_edge_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=decoder_layers,
                activation=decoder_activation,
                bias=False,
                device=self.device,
            )

        # ========== 1D Decoders (node + edge) ==========
        if self.with_decoder:
            self.node_decoder_1d = make_mlp(
                input_size=hidden_features,
                output_size=output_1d_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=decoder_layers,
                activation=decoder_activation,
                bias=False,
                device=self.device,
            )
            self.edge_decoder_1d = make_mlp(
                input_size=hidden_features,
                output_size=output_1d_edge_features,
                hidden_size=encoder_decoder_hidden,
                num_layers=decoder_layers,
                activation=decoder_activation,
                bias=False,
                device=self.device,
            )

    def _make_gnn(
        self,
        input_node_size: int,
        input_edge_size: int,
        output_node_size: int,
        output_edge_size: int,
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
                edge_out_channels=output_edge_size,
                hidden_size=hidden_features,
                num_layers=mlp_layers,
                activation=activation,
                residual=residual,
                edge_residual_scale=self.edge_residual_scale,
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
                        edge_out_channels=output_edge_size,
                        hidden_size=hidden_features,
                        num_layers=mlp_layers,
                        activation=activation,
                        residual=residual,
                        edge_residual_scale=self.edge_residual_scale,
                        bias=False,
                        device=device,
                    ),
                    "x, edge_index, edge_attr -> x, edge_attr",
                )
            )
        return PygSequential("x, edge_index, edge_attr", layers)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        x_1d: Tensor,
        edge_index_1d: Tensor,
        edge_attr_1d: Tensor,
        edge_index_1d_2d: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass for joint water-level (node) and discharge (edge) prediction.

        Returns:
            x_2d:        2D node water-level deltas   [num_nodes_2d, output_features]
            edge_attr_2d 2D edge discharge deltas     [num_edges_2d, output_edge_features]
            x_1d:        1D node water-level deltas    [num_nodes_1d, output_1d_features]
            edge_attr_1d 1D edge discharge deltas      [num_edges_1d, output_1d_edge_features]
        """
        # ========== Encoding ==========
        if self.with_encoder:
            x = self.node_encoder_2d(x)
            edge_attr = self.edge_encoder_2d(edge_attr)
            x_1d = self.node_encoder_1d(x_1d)
            edge_attr_1d = self.edge_encoder_1d(edge_attr_1d)

        # ========== 2D Message Passing ==========
        x_2d, edge_attr_2d = self.convs_2d(x, edge_index, edge_attr)

        # Apply normalization after GNN
        if self.use_layer_norm:
            x_2d = self.norm_2d(x_2d)

        # ========== 1D Message Passing ==========
        x_1d, edge_attr_1d = self.convs_1d(x_1d, edge_index_1d, edge_attr_1d)

        # Apply normalization after GNN
        if self.use_layer_norm:
            x_1d = self.norm_1d(x_1d)

        # ========== Coupling (1D <-> 2D) with Controlled Aggregation ==========
        if self.with_coupling:
            # ===== 1D -> 2D Coupling =====
            coupling_1d_feats = self.coupling_1d_to_2d(x_1d[edge_index_1d_2d[0]])
            coupling_to_2d = scatter_mean(
                coupling_1d_feats, edge_index_1d_2d[1], dim=0, dim_size=x_2d.size(0)
            )
            if self.use_coupling_gate:
                gate_input = torch.cat([x_2d, coupling_to_2d], dim=-1)
                gate = self.gate_1d_to_2d(gate_input)
                x_2d = x_2d + gate * coupling_to_2d
            else:
                x_2d = x_2d + coupling_to_2d
            if self.use_layer_norm:
                x_2d = self.norm_2d_post_coupling(x_2d)

            # ===== 2D -> 1D Coupling =====
            coupling_2d_feats = self.coupling_2d_to_1d(x_2d[edge_index_1d_2d[1]])
            coupling_to_1d = scatter_mean(
                coupling_2d_feats, edge_index_1d_2d[0], dim=0, dim_size=x_1d.size(0)
            )
            if self.use_coupling_gate:
                gate_input = torch.cat([x_1d, coupling_to_1d], dim=-1)
                gate = self.gate_2d_to_1d(gate_input)
                x_1d = x_1d + gate * coupling_to_1d
            else:
                x_1d = x_1d + coupling_to_1d
            if self.use_layer_norm:
                x_1d = self.norm_1d_post_coupling(x_1d)

        # ========== Decoding ==========
        if self.with_decoder:
            x_2d = self.node_decoder_2d(x_2d)
            edge_attr_2d = self.edge_decoder_2d(edge_attr_2d)
            x_1d = self.node_decoder_1d(x_1d)
            edge_attr_1d = self.edge_decoder_1d(edge_attr_1d)

            if self.clip_edge_predictions and self.max_edge_value is not None:
                edge_attr_2d = torch.clamp(
                    edge_attr_2d, -self.max_edge_value, self.max_edge_value
                )
                edge_attr_1d = torch.clamp(
                    edge_attr_1d, -self.max_edge_value, self.max_edge_value
                )

        return x_2d, edge_attr_2d, x_1d, edge_attr_1d


class NodeEdgeConv(MessagePassing):
    """
    Message  = MLP(node_i, edge_attr, node_j)   -> also the updated edge feature
    Aggregate = sum
    Node Update = MLP(aggregated_message) (+ residual node)
    Edge Update = message (+ scaled residual edge)
    """

    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        node_out_channels: int,
        edge_out_channels: int,
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

        # Message / edge-update MLP: outputs edge feature (also used as message)
        self.msg_mlp = make_mlp(
            input_size=2 * node_in_channels + edge_in_channels,
            output_size=edge_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            device=device,
        )

        # Node-update MLP: maps aggregated messages to node output
        self.node_mlp = make_mlp(
            input_size=node_in_channels,
            output_size=node_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            device=device,
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def propagate(self, edge_index, **kwargs):
        mutable_size = self._check_input(edge_index, size=None)
        coll_dict = self._collect(self._user_args, edge_index, mutable_size, kwargs)

        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        msg = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        aggr = self.aggregate(msg, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data("update", coll_dict)
        out = self.update(aggr, **update_kwargs)

        # Return updated node features and updated edge features (the message)
        return out, msg

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:
        cat_feats = torch.cat([x_i, edge_attr, x_j], dim=-1)
        msg = self.msg_mlp(cat_feats)
        if self.residual:
            # Scale down the edge residual to prevent edge value explosion
            msg = msg + self.edge_residual_scale * edge_attr
        return msg

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        out = self.node_mlp(aggr)
        if self.residual:
            out = out + x
        return out
