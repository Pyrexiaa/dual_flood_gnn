import torch
from torch.nn import LayerNorm
from torch_geometric.nn import MessagePassing
from utils.model_utils import make_mlp

from .base_model_1d2d import BaseModel1D2D
from torch_geometric.nn import HeteroConv


class DUALFloodHGNN1D2D(BaseModel1D2D):
    """
    Autoregressive heterogeneous GNN for coupled 1D-2D flood prediction.
    """

    def __init__(
        self,
        hidden_features: int,
        num_layers: int = 3,
        activation: str = "relu",
        **base_model_kwargs,
    ):
        super().__init__(**base_model_kwargs)

        self.hidden_features = hidden_features

        # ========== Input projections ==========
        self.node_proj_2d = make_mlp(
            self.input_node_features,
            hidden_features,
            hidden_features,
            num_layers=1,
            activation=activation,
            device=self.device,
        )

        self.node_proj_1d = make_mlp(
            self.input_1d_node_features,
            hidden_features,
            hidden_features,
            num_layers=1,
            activation=activation,
            device=self.device,
        )

        self.edge_proj_2d = make_mlp(
            self.input_edge_features,
            hidden_features,
            hidden_features,
            num_layers=1,
            activation=activation,
            device=self.device,
        )

        self.edge_proj_1d = make_mlp(
            self.input_1d_edge_features,
            hidden_features,
            hidden_features,
            num_layers=1,
            activation=activation,
            device=self.device,
        )

        # ========== HGNN layers ==========
        self.layers = torch.nn.ModuleList(
            [
                DualScaleHGNNLayer(hidden_features, activation, self.device)
                for _ in range(num_layers)
            ]
        )

        # ========== Output heads ==========
        self.out_2d = make_mlp(
            hidden_features,
            self.output_node_features,
            hidden_features,
            num_layers=2,
            activation=activation,
            device=self.device,
        )

        self.out_1d = make_mlp(
            hidden_features,
            self.output_1d_node_features,
            hidden_features,
            num_layers=2,
            activation=activation,
            device=self.device,
        )

    def forward(self, data):
        """
        Args:
            data: torch_geometric.data.HeteroData
                  containing autoregressive node/edge features

        Returns:
            x_2d_next: [N2, 1]
            x_1d_next: [N1, 1]
        """

        # ========== Project autoregressive inputs ==========
        x_dict = {
            "2d": self.node_proj_2d(data["2d"].x),
            "1d": self.node_proj_1d(data["1d"].x),
        }

        edge_attr_dict = {
            ("2d", "connects", "2d"): self.edge_proj_2d(
                data[("2d", "connects", "2d")].edge_attr
            ),
            ("1d", "connects", "1d"): self.edge_proj_1d(
                data[("1d", "connects", "1d")].edge_attr
            ),
            # coupling edges usually have no edge_attr
        }

        edge_index_dict = data.edge_index_dict

        # ========== HGNN propagation ==========
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        # ========== Predict next timestep ==========
        x_2d_next = self.out_2d(x_dict["2d"])
        x_1d_next = self.out_1d(x_dict["1d"])

        return {
            "2d": x_2d_next,
            "1d": x_1d_next,
        }


class DualScaleHGNNLayer(torch.nn.Module):
    def __init__(self, hidden_dim, activation="relu", device="cpu"):
        super().__init__()

        self.conv = HeteroConv(
            {
                ("1d", "connects", "1d"): NodeEdgeConv(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    device=device,
                ),
                ("2d", "connects", "2d"): NodeEdgeConv(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    device=device,
                ),
                ("1d", "couples", "2d"): NodeEdgeConv(
                    hidden_dim,
                    0,
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    device=device,
                ),
                ("2d", "couples", "1d"): NodeEdgeConv(
                    hidden_dim,
                    0,
                    hidden_dim,
                    hidden_dim,
                    activation=activation,
                    device=device,
                ),
            },
            aggr="mean",  # replaces scatter_mean
        )

        self.norm_1d = LayerNorm(hidden_dim)
        self.norm_2d = LayerNorm(hidden_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv(
            x_dict,
            edge_index_dict,
            edge_attr_dict,
        )

        x_dict["1d"] = self.norm_1d(x_dict["1d"])
        x_dict["2d"] = self.norm_2d(x_dict["2d"])

        return x_dict


class NodeEdgeConv(MessagePassing):
    def __init__(
        self,
        node_in_channels,
        edge_in_channels,
        node_out_channels,
        hidden_size,
        num_layers=2,
        activation="relu",
        residual=True,
        edge_residual_scale=0.2,
        bias=False,
        device="cpu",
    ):
        super().__init__(aggr="sum")
        self.residual = residual
        self.edge_residual_scale = edge_residual_scale
        self.edge_in_channels = edge_in_channels

        msg_in = 2 * node_in_channels + (edge_in_channels or 0)

        self.msg_mlp = make_mlp(
            input_size=msg_in,
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

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            feats = torch.cat([x_i, edge_attr, x_j], dim=-1)
        else:
            feats = torch.cat([x_i, x_j], dim=-1)

        msg = self.msg_mlp(feats)

        if self.residual and edge_attr is not None:
            msg = msg + self.edge_residual_scale * edge_attr

        return msg

    def update(self, aggr, x):
        # Handle hetero (bipartite) case
        # PyG Convention x = (x_src, x_dst)
        if isinstance(x, tuple):
            x_dst = x[1]
        else:
            x_dst = x

        out = self.node_mlp(aggr)

        if self.residual:
            out = out + x_dst

        return out

