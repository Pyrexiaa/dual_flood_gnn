from .autoregressive_flood_1d2d_dataset import AutoregressiveFlood1D2DDataset
from .in_memory_flood_1d2d_dataset import InMemoryFlood1D2DDataset

class InMemoryAutoregressiveFlood1D2DDataset(AutoregressiveFlood1D2DDataset, InMemoryFlood1D2DDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
