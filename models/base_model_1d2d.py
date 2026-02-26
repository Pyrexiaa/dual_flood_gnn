from torch import Tensor
from torch.nn import Module

class BaseModel1D2D(Module):
    def __init__(self,
                 static_node_features: int,
                 dynamic_node_features: int,
                 static_edge_features: int,
                 dynamic_edge_features: int,
                 static_1d_node_features: int,
                 dynamic_1d_node_features: int,
                 static_1d_edge_features: int,
                 dynamic_1d_edge_features: int,
                 previous_timesteps: int,
                 input_align_features: int = None,
                 input_align_edge_features: int = None,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.previous_timesteps = previous_timesteps
        self.input_align_features = input_align_features
        self.input_align_edge_features = input_align_edge_features
        
        # 2D features
        self.input_node_features = static_node_features + (dynamic_node_features * (previous_timesteps + 1))
        self.output_node_features = 1
        self.input_edge_features = static_edge_features + (dynamic_edge_features * (previous_timesteps + 1))
        self.output_edge_features = 1
        
        # 1D features
        self.input_1d_node_features = static_1d_node_features + (dynamic_1d_node_features * (previous_timesteps + 1))
        self.output_1d_node_features = 1
        self.input_1d_edge_features = static_1d_edge_features + (dynamic_1d_edge_features * (previous_timesteps + 1))
        self.output_1d_edge_features = 1

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        raise NotImplementedError("Forward method not implemented!")

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
