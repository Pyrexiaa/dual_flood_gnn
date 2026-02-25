import os
import numpy as np
import torch
from torch.nn import Module
from data import FloodEvent1D2DDataset
from utils import Logger
from utils.validation_stats import ValidationStats
from typing import List, Optional

class Base1D2DTester:
    def __init__(self,
                 model: Module,
                 dataset: FloodEvent1D2DDataset,
                 rollout_start: int = 0,
                 rollout_timesteps: Optional[int] = None,
                 include_physics_loss: bool = True,
                 logger: Logger = None,
                 feature_alignment: str = None,
                 device: str = 'cpu'):
        self.model = model
        self.dataset = dataset
        self.rollout_start = rollout_start
        self.rollout_timesteps = rollout_timesteps
        self.include_physics_loss = include_physics_loss
        self.logger = logger
        self.device = device
        self.events_validation_stats: List[ValidationStats] = []
        self.feature_alignment = feature_alignment

        self.log = print
        if logger is not None and hasattr(logger, 'log'):
            self.log = logger.log

        # Get non-boundary nodes/edges and threshold for metric computation
        self.boundary_nodes_mask = dataset.boundary_condition.boundary_nodes_mask
        self.non_boundary_nodes_mask = ~dataset.boundary_condition.boundary_nodes_mask
        self.boundary_edges_mask = dataset.boundary_condition.boundary_edges_mask

        # ===== 2D Threshold Setup =====
        elevation_2d_idx = dataset.STATIC_NODE_FEATURES.index('elevation')
        self.ground_elevation_2d = dataset[0].x.clone()[:, elevation_2d_idx]

        if dataset.is_normalized:
            self.ground_elevation_2d = dataset.normalizer.denormalize('elevation', self.ground_elevation_2d)

        self.ground_elevation_2d = self.ground_elevation_2d[:, None] # Add dimension for consistency

        # 2D depth threshold (e.g., 0.16 ft = ~5 cm)
        self.depth_threshold_2d = 0.16  # feet
        self.threshold_per_cell_2d = torch.full_like(self.ground_elevation_2d, self.depth_threshold_2d)

        # ===== 1D Threshold Setup =====
        surface_elevation_1d_idx = dataset.STATIC_1D_NODE_FEATURES.index('surface_elevation')
        self.surface_elevation_1d = dataset[0].x_1d.clone()[:, surface_elevation_1d_idx]

        if dataset.is_normalized:
            self.surface_elevation_1d = dataset.normalizer.denormalize('surface_elevation', self.surface_elevation_1d)

        self.surface_elevation_1d = self.surface_elevation_1d[:, None]  # Add dimension for consistency

        # 1D depth threshold (might want different threshold for drainage)
        self.depth_threshold_1d = 0.16  # feet (or use 0.33 for more conservative)
        self.threshold_per_cell_1d = torch.full_like(self.surface_elevation_1d, self.depth_threshold_1d)

        # Get sliding window indices
        previous_timesteps = dataset.previous_timesteps
        sliding_window_length = previous_timesteps + 1

        target_nodes_idx = dataset.DYNAMIC_NODE_FEATURES.index(dataset.NODE_TARGET_FEATURE)
        self.start_node_target_idx = dataset.num_static_node_features + (target_nodes_idx * sliding_window_length)
        self.end_node_target_idx = self.start_node_target_idx + sliding_window_length

        target_edges_idx = dataset.DYNAMIC_EDGE_FEATURES.index(dataset.EDGE_TARGET_FEATURE)
        self.start_edge_target_idx = dataset.num_static_edge_features + (target_edges_idx * sliding_window_length)
        self.end_edge_target_idx = self.start_edge_target_idx + sliding_window_length

        target_1d_nodes_idx = dataset.DYNAMIC_1D_NODE_FEATURES.index(dataset.NODE_1D_TARGET_FEATURE)
        self.start_1d_node_target_idx = dataset.num_static_1d_node_features + (target_1d_nodes_idx * sliding_window_length)
        self.end_1d_node_target_idx = self.start_1d_node_target_idx + sliding_window_length

        target_1d_edges_idx = dataset.DYNAMIC_1D_EDGE_FEATURES.index(dataset.EDGE_1D_TARGET_FEATURE)
        self.start_1d_edge_target_idx = dataset.num_static_1d_edge_features + (target_1d_edges_idx * sliding_window_length)
        self.end_1d_edge_target_idx = self.start_1d_edge_target_idx + sliding_window_length

    def test(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_avg_node_rmse(self) -> float:
        rmses = [stat.get_avg_rmse() for stat in self.events_validation_stats]
        return np.mean(rmses) if rmses else 0.0

    def get_avg_node_mae(self) -> float:
        maes = [stat.get_avg_mae() for stat in self.events_validation_stats]
        return np.mean(maes) if maes else 0.0

    def get_avg_node_nse(self) -> float:
        nses = [stat.get_avg_nse() for stat in self.events_validation_stats]
        return np.mean(nses) if nses else 0.0

    def get_avg_edge_rmse(self) -> float:
        edge_rmses = [stat.get_avg_edge_rmse() for stat in self.events_validation_stats]
        return np.mean(edge_rmses) if edge_rmses else 0.0

    def get_avg_edge_mae(self) -> float:
        edge_maes = [stat.get_avg_edge_mae() for stat in self.events_validation_stats]
        return np.mean(edge_maes) if edge_maes else 0.0

    def get_avg_edge_nse(self) -> float:
        edge_nses = [stat.get_avg_edge_nse() for stat in self.events_validation_stats]
        return np.mean(edge_nses) if edge_nses else 0.0
    
    def get_avg_node_1d_rmse(self) -> float:
        rmses = [stat.get_avg_1d_rmse() for stat in self.events_validation_stats]
        return np.mean(rmses) if rmses else 0.0

    def get_avg_node_1d_mae(self) -> float:
        maes = [stat.get_avg_1d_mae() for stat in self.events_validation_stats]
        return np.mean(maes) if maes else 0.0

    def get_avg_node_1d_nse(self) -> float:
        nses = [stat.get_avg_1d_nse() for stat in self.events_validation_stats]
        return np.mean(nses) if nses else 0.0

    def get_avg_edge_1d_rmse(self) -> float:
        rmses = [stat.get_avg_1d_edge_rmse() for stat in self.events_validation_stats]
        return np.mean(rmses) if rmses else 0.0

    def get_avg_edge_1d_mae(self) -> float:
        maes = [stat.get_avg_1d_edge_mae() for stat in self.events_validation_stats]
        return np.mean(maes) if maes else 0.0

    def get_avg_edge_1d_nse(self) -> float:
        nses = [stat.get_avg_1d_edge_nse() for stat in self.events_validation_stats]
        return np.mean(nses) if nses else 0.0

    def get_avg_abs_global_mass_loss(self) -> float:
        losses = [abs(stat.get_total_global_mass_loss()) for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def get_avg_abs_local_mass_loss(self) -> float:
        losses = [abs(stat.get_total_local_mass_loss()) for stat in self.events_validation_stats]
        return np.mean(losses) if losses else 0.0

    def save_stats(self, output_dir: str, stats_filename_prefix: Optional[str] = None):
        for event_idx, run_id in enumerate(self.dataset.hec_ras_run_ids):
            validation_stats = self.events_validation_stats[event_idx]
            saved_metrics_path = os.path.join(output_dir, f'{stats_filename_prefix}_runid_{run_id}_test_metrics.npz')
            validation_stats.save_stats(saved_metrics_path)
