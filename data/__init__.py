from typing import Literal

from .autoregressive_flood_dataset import AutoregressiveFloodDataset
from .flood_event_dataset import FloodEventDataset
from .flood_event_1d2d_dataset import FloodEvent1D2DDataset
from .in_memory_autoregressive_flood_dataset import InMemoryAutoregressiveFloodDataset
from .in_memory_flood_dataset import InMemoryFloodDataset
from .in_memory_flood_1d2d_dataset import InMemoryFlood1D2DDataset

def dataset_factory(storage_mode: Literal['memory', 'disk'], autoregressive: bool, *args, **kwargs) -> FloodEvent1D2DDataset:
    if autoregressive:
        if storage_mode == 'memory':
            return InMemoryAutoregressiveFloodDataset(*args, **kwargs)
        elif storage_mode == 'disk':
            return AutoregressiveFloodDataset(*args, **kwargs)

    if storage_mode == 'memory':
        return InMemoryFlood1D2DDataset(*args, **kwargs)
    elif storage_mode == 'disk':
        return FloodEvent1D2DDataset(*args, **kwargs)

    raise ValueError(f'Dataset class is not defined.')

__all__ = [
    'AutoregressiveFloodDataset',
    'FloodEvent1D2DDataset',
    'InMemoryAutoregressiveFloodDataset',
    'InMemoryFlood1D2DDataset',
    'dataset_factory',
]
