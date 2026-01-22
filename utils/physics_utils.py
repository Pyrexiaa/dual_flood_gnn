from data import FloodEvent1D2DDataset
import torch
from torch import Tensor
from typing import Optional

# =============== Combined Functions ===============

def get_physics_info_node_edge(x: Tensor, edge_attr: Tensor, previous_timesteps: int, batch):
    curr_water_volume = get_curr_volume_from_node_features(x, previous_timesteps)
    curr_face_flow = get_curr_flow_from_edge_features(edge_attr, previous_timesteps)
    return curr_water_volume, curr_face_flow

# =============== Individual Functions ===============

# def get_curr_volume_from_node_features(x: Tensor, previous_timesteps: int) -> Tensor:
#     water_volume_dyn_num = FloodEvent1D2DDataset.DYNAMIC_NODE_FEATURES.index('water_volume') + 1
#     num_static_node_features = len(FloodEvent1D2DDataset.STATIC_NODE_FEATURES)
#     curr_water_volume_idx = num_static_node_features + ((previous_timesteps + 1) * water_volume_dyn_num) - 1
#     curr_water_volume = x[:, [curr_water_volume_idx]]
#     return curr_water_volume

def get_curr_volume_from_node_features(x: Tensor, previous_timesteps: int) -> Tensor:
    """
    Compute current water volume from water level and cell area.
    
    Water volume = water level * area
    """
    water_level_dyn_num = FloodEvent1D2DDataset.DYNAMIC_NODE_FEATURES.index('water_level') + 1
    num_static_node_features = len(FloodEvent1D2DDataset.STATIC_NODE_FEATURES)
    curr_water_level_idx = num_static_node_features + ((previous_timesteps + 1) * water_level_dyn_num) - 1
    curr_water_level = x[:, [curr_water_level_idx]]
    area_idx = FloodEvent1D2DDataset.STATIC_NODE_FEATURES.index('area')
    node_areas = x[:, area_idx:area_idx+1]
    curr_water_volume = curr_water_level * node_areas
    
    return curr_water_volume

def get_curr_flow_from_edge_features(edge_attr: Tensor, previous_timesteps: int) -> Tensor:
    flow_dyn_num = FloodEvent1D2DDataset.DYNAMIC_EDGE_FEATURES.index('flow') + 1
    num_static_edge_features = len(FloodEvent1D2DDataset.STATIC_EDGE_FEATURES)
    curr_flow_idx = num_static_edge_features + ((previous_timesteps + 1) * flow_dyn_num) - 1
    curr_flow = edge_attr[:, [curr_flow_idx]]
    return curr_flow

def get_total_rainfall(batch, current_timestep: Optional[int] = None):
    assert hasattr(batch, 'global_mass_info'), "Global mass conservation data must be included in the dataset"
    total_rainfall = batch.global_mass_info['total_rainfall']
    if current_timestep is None:
        return total_rainfall
    assert len(total_rainfall.shape) == 2, "Current timestep can only be specified for per-timestep rainfall data from autoregressive datasets"
    if isinstance(total_rainfall, torch.Tensor) and total_rainfall.dtype == torch.float64:
        total_rainfall = total_rainfall.float()
    return total_rainfall[:, current_timestep]

def get_rainfall(batch, current_timestep: Optional[int] = None):
    assert hasattr(batch, 'local_mass_info'), "Local mass conservation data must be included in the dataset"
    rainfall = batch.local_mass_info['rainfall']
    if current_timestep is None:
        return rainfall
    assert len(rainfall.shape) == 2, "Current timestep can only be specified for per-timestep rainfall data from autoregressive datasets"
    if isinstance(rainfall, torch.Tensor) and rainfall.dtype == torch.float64:
        rainfall = rainfall.float()
    return rainfall[:, current_timestep]
