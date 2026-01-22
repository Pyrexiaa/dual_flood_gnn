from torch_geometric.data import Data
import torch

class Data1D2D(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes
        elif key == 'edge_index_1d':
            return self.num_nodes_1d
        elif key == 'edge_index_1d_2d':
            return torch.tensor([[self.num_nodes_1d], [self.num_nodes]])
        else:
            return super().__inc__(key, value, *args, **kwargs)