import torch
from torch import Tensor
from torch.nn import LayerNorm
from torch_geometric.nn import MessagePassing, Sequential as PygSequential
from typing import Tuple
from utils.model_utils import make_mlp
from torch_scatter import scatter_mean

from .base_model_1d2d import BaseModel1D2D


class DUALFloodGNNNode1D2D(BaseModel1D2D):
    """
    No edge features - only node features and predictions.
    """

    def __init__(
        self,
        # 2D parameters
        input_features: int = None,
        output_features: int = None,
        # 1D parameters
        input_1d_features: int = None,
        output_1d_features: int = None,
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
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.with_coupling = coupling_layers > 0
        self.use_coupling_gate = use_coupling_gate
        self.use_layer_norm = use_layer_norm

        # Set default values from base model
        if input_features is None:
            input_features = self.input_node_features
        if output_features is None:
            output_features = self.output_node_features

        if input_1d_features is None:
            input_1d_features = self.input_1d_node_features
        if output_1d_features is None:
            output_1d_features = self.output_1d_node_features

        if coupling_hidden is None:
            coupling_hidden = hidden_features

        encoder_decoder_hidden = hidden_features * 2

        # ========== 2D Encoder ==========
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

        # ========== 1D Encoder ==========
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

        # Determine input/output sizes for GNN layers
        input_node_size_2d = hidden_features if self.with_encoder else input_features
        output_node_size_2d = hidden_features if self.with_decoder else output_features

        input_node_size_1d = hidden_features if self.with_encoder else input_1d_features
        output_node_size_1d = (
            hidden_features if self.with_decoder else output_1d_features
        )

        # ========== 2D GNN Layers ==========
        self.convs_2d = self._make_gnn(
            input_node_size=input_node_size_2d,
            output_node_size=output_node_size_2d,
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
            output_node_size=output_node_size_1d,
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

        # ========== 2D Decoder ==========
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

        # ========== 1D Decoder ==========
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

    def _make_gnn(
        self,
        input_node_size: int,
        output_node_size: int,
        hidden_features: int,
        num_layers: int,
        mlp_layers: int,
        activation: str,
        residual: bool,
        device: str,
    ):
        if num_layers == 1:
            return NodeConv(
                node_in_channels=input_node_size,
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
                    NodeConv(
                        node_in_channels=input_node_size,
                        node_out_channels=output_node_size,
                        hidden_size=hidden_features,
                        num_layers=mlp_layers,
                        activation=activation,
                        residual=residual,
                        bias=False,
                        device=device,
                    ),
                    "x, edge_index -> x",
                )
            )
        return PygSequential("x, edge_index", layers)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        x_1d: Tensor,
        edge_index_1d: Tensor,
        edge_index_1d_2d: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for water level prediction.

        Args:
            x: 2D node features [num_nodes_2d, input_features]
            edge_index: 2D edge connectivity [2, num_edges_2d]
            x_1d: 1D node features [num_nodes_1d, input_1d_features]
            edge_index_1d: 1D edge connectivity [2, num_edges_1d]
            edge_index_1d_2d: Coupling edge connectivity [2, num_coupling_edges]

        Returns:
            x_2d: Predicted water levels for 2D nodes [num_nodes_2d, output_features]
            x_1d: Predicted water levels for 1D nodes [num_nodes_1d, output_1d_features]
        """
        # ========== Encoding ==========
        if self.with_encoder:
            x = self.node_encoder_2d(x)
            x_1d = self.node_encoder_1d(x_1d)

        # ========== 2D Message Passing ==========
        x_2d = self.convs_2d(x, edge_index)

        # Apply normalization after GNN
        if self.use_layer_norm:
            x_2d = self.norm_2d(x_2d)

        # ========== 1D Message Passing ==========
        x_1d = self.convs_1d(x_1d, edge_index_1d)

        # Apply normalization after GNN
        if self.use_layer_norm:
            x_1d = self.norm_1d(x_1d)

        # ========== Coupling (1D <-> 2D) with Controlled Aggregation ==========
        if self.with_coupling:
            # ===== 1D -> 2D Coupling =====
            # Transform 1D features for coupling
            coupling_1d_feats = self.coupling_1d_to_2d(x_1d[edge_index_1d_2d[0]])

            # Use scatter_mean to prevent accumulation
            coupling_to_2d = scatter_mean(
                coupling_1d_feats, edge_index_1d_2d[1], dim=0, dim_size=x_2d.size(0)
            )

            # Apply gating mechanism
            if self.use_coupling_gate:
                gate_input = torch.cat([x_2d, coupling_to_2d], dim=-1)
                gate = self.gate_1d_to_2d(gate_input)
                x_2d = x_2d + gate * coupling_to_2d
            else:
                x_2d = x_2d + coupling_to_2d

            # Normalize after coupling
            if self.use_layer_norm:
                x_2d = self.norm_2d_post_coupling(x_2d)

            # ===== 2D -> 1D Coupling =====
            # Transform 2D features for coupling
            coupling_2d_feats = self.coupling_2d_to_1d(x_2d[edge_index_1d_2d[1]])

            # Use scatter_mean
            coupling_to_1d = scatter_mean(
                coupling_2d_feats, edge_index_1d_2d[0], dim=0, dim_size=x_1d.size(0)
            )

            # Apply gating mechanism - decide which signal from 2d nodes are useful for 1d node
            if self.use_coupling_gate:
                gate_input = torch.cat([x_1d, coupling_to_1d], dim=-1)
                gate = self.gate_2d_to_1d(gate_input) # sigmoid probability
                x_1d = x_1d + gate * coupling_to_1d
            else:
                x_1d = x_1d + coupling_to_1d

            # Normalize after coupling
            if self.use_layer_norm:
                x_1d = self.norm_1d_post_coupling(x_1d)

        # ========== Decoding ==========
        if self.with_decoder:
            x_2d = self.node_decoder_2d(x_2d)
            x_1d = self.node_decoder_1d(x_1d)

        return x_2d, x_1d


class NodeConv(MessagePassing):
    """
    Message passing for node-only updates.
    Message = MLP(node_j)
    Aggregate = sum
    Node Update = MLP(node_i, aggregated_message)
    """

    def __init__(
        self,
        node_in_channels: int,
        node_out_channels: int,
        hidden_size: int,
        num_layers: int = 2,
        activation: str = "relu",
        residual: bool = True,
        bias: bool = False,
        device: str = "cpu",
    ):
        super().__init__(aggr="sum")
        self.residual = residual

        # MLP for message computation (from neighbor nodes)
        self.msg_mlp = make_mlp(
            input_size=node_in_channels,
            output_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            device=device,
        )

        # MLP for node update (combine current node with aggregated messages)
        input_size = node_in_channels + hidden_size
        output_size = node_out_channels
        self.node_mlp = make_mlp(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            bias=bias,
            device=device,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        """Compute messages from neighbor nodes"""
        return self.msg_mlp(x_j)

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        """Update node features by combining with aggregated messages"""
        combined = torch.cat([x, aggr], dim=-1)
        out = self.node_mlp(combined)

        if self.residual:
            out = out + x

        return out
