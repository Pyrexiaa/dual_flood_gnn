import torch

from constants import TEST_LOCAL_MASS_LOSS_NODES
from torch_geometric.loader import DataLoader
from utils.validation_stats import ValidationStats


from .base_1d2d_tester import Base1D2DTester


class DualAutoregressive1D2DTester(Base1D2DTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self):
        for event_idx, run_id in enumerate(self.dataset.hec_ras_run_ids):
            self.log(
                f"Validating on run {event_idx + 1}/{len(self.dataset.hec_ras_run_ids)} with Run ID {run_id}"
            )

            validation_stats = ValidationStats(
                logger=self.logger,
                normalizer=self.dataset.normalizer,
                is_normalized=self.dataset.is_normalized,
                delta_t=self.dataset.timestep_interval,
            )
            self.run_test_for_event(event_idx, validation_stats)
            validation_stats.print_stats_summary()
            self.events_validation_stats.append(validation_stats)

        # 2D Metrics
        self.log("Average 2D NODE METRICS across events:")
        self.log(f"\tRMSE: {self.get_avg_node_rmse():.4e}")
        self.log(f"\tMAE: {self.get_avg_node_mae():.4e}")
        self.log(f"\tNSE: {self.get_avg_node_nse():.4e}")

        self.log("Average 2D EDGE METRICS across events:")
        self.log(f"\tRMSE: {self.get_avg_edge_rmse():.4e}")
        self.log(f"\tMAE: {self.get_avg_edge_mae():.4e}")
        self.log(f"\tNSE: {self.get_avg_edge_nse():.4e}")

        # 1D Metrics
        self.log("Average 1D NODE METRICS across events:")
        self.log(f"\tRMSE: {self.get_avg_node_1d_rmse():.4e}")
        self.log(f"\tMAE: {self.get_avg_node_1d_mae():.4e}")
        self.log(f"\tNSE: {self.get_avg_node_1d_nse():.4e}")

        self.log("Average 1D EDGE METRICS across events:")
        self.log(f"\tRMSE: {self.get_avg_edge_1d_rmse():.4e}")
        self.log(f"\tMAE: {self.get_avg_edge_1d_mae():.4e}")
        self.log(f"\tNSE: {self.get_avg_edge_1d_nse():.4e}")

        if self.include_physics_loss:
            self.log("PHYSICS-BASED METRICS across events:")
            self.log(
                f"\tAverage Absolute Global Mass Conservation Loss: {self.get_avg_abs_global_mass_loss():.4e}"
            )
            self.log(
                f"\tAverage Absolute Local Mass Conservation Loss: {self.get_avg_abs_local_mass_loss():.4e}"
            )

    def run_test_for_event(self, event_idx: int, validation_stats: ValidationStats):
        validation_stats.start_validate()
        self.model.eval()
        with torch.no_grad():
            event_start_idx = (
                self.dataset.event_start_idx[event_idx] + self.rollout_start
            )
            event_end_idx = (
                self.dataset.event_start_idx[event_idx + 1]
                if event_idx + 1 < len(self.dataset.event_start_idx)
                else self.dataset.total_rollout_timesteps
            )
            if self.rollout_timesteps is not None:
                event_end_idx = event_start_idx + self.rollout_timesteps
                dataset_event_length = (
                    self.dataset.event_start_idx[event_idx + 1]
                    if event_idx + 1 < len(self.dataset.event_start_idx)
                    else self.dataset.total_rollout_timesteps
                )
                assert event_end_idx <= dataset_event_length, (
                    f"Rollout length {event_end_idx} exceeds event length {dataset_event_length} for event {self.dataset.hec_ras_run_ids[event_idx]}."
                )
            event_dataset = self.dataset[event_start_idx:event_end_idx]
            dataloader = DataLoader(event_dataset, batch_size=1, shuffle=False)

            # 2D sliding windows
            sliding_window = (
                self.dataset[event_start_idx]
                .x[:, self.start_node_target_idx : self.end_node_target_idx]
                .clone()
            )
            edge_sliding_window = (
                self.dataset[event_start_idx]
                .edge_attr[:, self.start_edge_target_idx : self.end_edge_target_idx]
                .clone()
            )

            # 1D sliding windows
            sliding_window_1d = (
                self.dataset[event_start_idx]
                .x_1d[:, self.start_1d_node_target_idx : self.end_1d_node_target_idx]
                .clone()
            )
            edge_sliding_window_1d = (
                self.dataset[event_start_idx]
                .edge_attr_1d[
                    :, self.start_1d_edge_target_idx : self.end_1d_edge_target_idx
                ]
                .clone()
            )

            # Move to device
            sliding_window, edge_sliding_window = (
                sliding_window.to(self.device),
                edge_sliding_window.to(self.device),
            )
            sliding_window_1d, edge_sliding_window_1d = (
                sliding_window_1d.to(self.device),
                edge_sliding_window_1d.to(self.device),
            )

            for graph in dataloader:
                graph = graph.to(self.device)

                graph.x = graph.x.float()
                graph.edge_attr = graph.edge_attr.float()
                graph.y = graph.y.float()
                graph.y_edge = graph.y_edge.float()
                graph.x_1d = graph.x_1d.float()
                graph.edge_attr_1d = graph.edge_attr_1d.float()
                graph.y_1d = graph.y_1d.float()
                graph.y_1d_edge = graph.y_1d_edge.float()

                part1 = graph.x[:, : self.start_node_target_idx]
                part2 = sliding_window
                part3 = graph.x[:, self.end_node_target_idx :]
                
                # 2D features
                x = torch.concat([part1, part2, part3], dim=1)
                edge_attr = torch.concat(
                    [
                        graph.edge_attr[:, : self.start_edge_target_idx],
                        edge_sliding_window,
                        graph.edge_attr[:, self.end_edge_target_idx :],
                    ],
                    dim=1,
                )
                edge_index = graph.edge_index

                # 1D features
                x_1d = torch.concat(
                    [
                        graph.x_1d[:, : self.start_1d_node_target_idx],
                        sliding_window_1d,
                        graph.x_1d[:, self.end_1d_node_target_idx :],
                    ],
                    dim=1,
                )
                edge_attr_1d = torch.concat(
                    [
                        graph.edge_attr_1d[:, : self.start_1d_edge_target_idx],
                        edge_sliding_window_1d,
                        graph.edge_attr_1d[:, self.end_1d_edge_target_idx :],
                    ],
                    dim=1,
                )
                edge_index_1d = graph.edge_index_1d

                # Model prediction - returns 4 outputs
                pred_diff, edge_pred_diff, pred_diff_1d, edge_pred_diff_1d = self.model(
                    x,
                    edge_index,
                    edge_attr,
                    x_1d,
                    edge_index_1d,
                    edge_attr_1d,
                    graph.edge_index_1d_2d,
                )

                # Override boundary conditions in 2D predictions
                pred_diff[self.boundary_nodes_mask] = graph.y[self.boundary_nodes_mask]
                edge_pred_diff[self.boundary_edges_mask] = graph.y_edge[
                    self.boundary_edges_mask
                ]

                # Override boundary conditions in 1D predictions (if applicable)
                if hasattr(self, "boundary_nodes_1d_mask"):
                    pred_diff_1d[self.boundary_nodes_1d_mask] = graph.y_1d[
                        self.boundary_nodes_1d_mask
                    ]
                if hasattr(self, "boundary_edges_1d_mask"):
                    edge_pred_diff_1d[self.boundary_edges_1d_mask] = graph.y_1d_edge[
                        self.boundary_edges_1d_mask
                    ]

                # 2D predictions
                prev_node_pred = sliding_window[:, [-1]]
                pred = prev_node_pred + pred_diff
                prev_edge_pred = edge_sliding_window[:, [-1]]
                edge_pred = prev_edge_pred + edge_pred_diff

                # 1D predictions
                prev_node_pred_1d = sliding_window_1d[:, [-1]]
                pred_1d = prev_node_pred_1d + pred_diff_1d
                prev_edge_pred_1d = edge_sliding_window_1d[:, [-1]]
                edge_pred_1d = prev_edge_pred_1d + edge_pred_diff_1d

                if self.include_physics_loss:
                    # You may need to update this to include 1D physics
                    validation_stats.update_physics_informed_stats_for_timestep(
                        pred,
                        prev_node_pred,
                        prev_edge_pred,
                        graph,
                        TEST_LOCAL_MASS_LOSS_NODES,
                    )

                # Update sliding windows
                sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)
                edge_sliding_window = torch.concat(
                    (edge_sliding_window[:, 1:], edge_pred), dim=1
                )
                sliding_window_1d = torch.concat(
                    (sliding_window_1d[:, 1:], pred_1d), dim=1
                )
                edge_sliding_window_1d = torch.concat(
                    (edge_sliding_window_1d[:, 1:], edge_pred_1d), dim=1
                )

                # ===== 2D Validation =====
                label = graph.x[:, [self.end_node_target_idx - 1]] + graph.y
                if self.dataset.is_normalized:
                    pred = self.dataset.normalizer.denormalize(
                        self.dataset.NODE_TARGET_FEATURE, pred
                    )
                    label = self.dataset.normalizer.denormalize(
                        self.dataset.NODE_TARGET_FEATURE, label
                    )

                pred = torch.clip(pred, min=0)
                label = torch.clip(label, min=0)

                pred = pred[self.non_boundary_nodes_mask]
                label = label[self.non_boundary_nodes_mask]

                validation_stats.update_stats_for_timestep(
                    pred.cpu(),
                    label.cpu(),
                    pred_ground_elevation=self.ground_elevation_2d[self.non_boundary_nodes_mask].cpu(),
                    label_ground_elevation=self.ground_elevation_2d[self.non_boundary_nodes_mask].cpu(),
                    water_threshold=self.threshold_per_cell_2d[self.non_boundary_nodes_mask],
                    timestamp=graph.timestep if hasattr(graph, "timestep") else None,
                )

                label_edge = (
                    graph.edge_attr[:, [self.end_edge_target_idx - 1]] + graph.y_edge
                )
                if self.dataset.is_normalized:
                    edge_pred = self.dataset.normalizer.denormalize(
                        self.dataset.EDGE_TARGET_FEATURE, edge_pred
                    )
                    label_edge = self.dataset.normalizer.denormalize(
                        self.dataset.EDGE_TARGET_FEATURE, label_edge
                    )

                validation_stats.update_edge_stats_for_timestep(
                    edge_pred.cpu(), label_edge.cpu()
                )

                # ===== 1D Validation =====
                label_1d = graph.x_1d[:, [self.end_1d_node_target_idx - 1]] + graph.y_1d
                if self.dataset.is_normalized:
                    pred_1d = self.dataset.normalizer.denormalize(
                        self.dataset.NODE_1D_TARGET_FEATURE, pred_1d
                    )
                    label_1d = self.dataset.normalizer.denormalize(
                        self.dataset.NODE_1D_TARGET_FEATURE, label_1d
                    )

                pred_1d = torch.clip(pred_1d, min=0)
                label_1d = torch.clip(label_1d, min=0)

                validation_stats.update_1d_stats_for_timestep(
                    pred_1d.cpu(),
                    label_1d.cpu(), 
                    pred_ground_elevation=self.surface_elevation_1d.cpu(),
                    label_ground_elevation=self.surface_elevation_1d.cpu(),
                    water_threshold=self.threshold_per_cell_1d,
                    timestamp=graph.timestep if hasattr(graph, "timestep") else None,
                )

                label_edge_1d = (
                    graph.edge_attr_1d[:, [self.end_1d_edge_target_idx - 1]]
                    + graph.y_1d_edge
                )
                if self.dataset.is_normalized:
                    edge_pred_1d = self.dataset.normalizer.denormalize(
                        self.dataset.EDGE_1D_TARGET_FEATURE, edge_pred_1d
                    )
                    label_edge_1d = self.dataset.normalizer.denormalize(
                        self.dataset.EDGE_1D_TARGET_FEATURE, label_edge_1d
                    )

                validation_stats.update_1d_edge_stats_for_timestep(
                    edge_pred_1d.cpu(), label_edge_1d.cpu()
                )

        validation_stats.end_validate()
