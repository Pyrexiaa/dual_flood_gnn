from pathlib import Path
import torch

from constants import TEST_LOCAL_MASS_LOSS_NODES
from torch_geometric.loader import DataLoader
from utils.validation_stats import ValidationStats
from utils import physics_utils
import pandas as pd

from .base_1d2d_tester import Base1D2DTester
from data.feature_aligner import BatchTensorAligner


class NodeEdgeAutoregressive1D2DTester(Base1D2DTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_aligner = BatchTensorAligner(self.dataset).to(self.device)

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
            self.run_test_for_event(event_idx, validation_stats, predictions_dir=f"./kaggle_submissions/{self.dataset.model_name}_{self.dataset.previous_timesteps}")
            validation_stats.print_stats_summary()
            self.events_validation_stats.append(validation_stats)

        # 2D Node metrics
        self.log(f"Average 2D Node RMSE across events: {self.get_avg_node_rmse():.4e}")
        self.log(f"Average 2D Node MAE across events: {self.get_avg_node_mae():.4e}")
        self.log(f"Average 2D Node NSE across events: {self.get_avg_node_nse():.4e}")

        # 1D Node metrics
        self.log(
            f"Average 1D Node RMSE across events: {self.get_avg_node_1d_rmse():.4e}"
        )
        self.log(f"Average 1D Node MAE across events: {self.get_avg_node_1d_mae():.4e}")
        self.log(f"Average 1D Node NSE across events: {self.get_avg_node_1d_nse():.4e}")

        if self.include_physics_loss:
            self.log(
                f"Average Absolute Global Mass Conservation Loss across events: {self.get_avg_abs_global_mass_loss():.4e}"
            )
            self.log(
                f"Average Absolute Local Mass Conservation Loss across events: {self.get_avg_abs_local_mass_loss():.4e}"
            )

    def run_test_for_event(
        self,
        event_idx: int,
        validation_stats: ValidationStats,
        save_predictions: bool = True,
        predictions_dir="./kaggle_submissions/model1"
    ):
        validation_stats.start_validate()
        self.model.eval()

        # Initialize predictions list if saving
        if save_predictions:
            predictions_list = []
            row_id = 0
            model_id = 1

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
                assert event_end_idx <= (
                    self.dataset.event_start_idx[event_idx + 1]
                    if event_idx + 1 < len(self.dataset.event_start_idx)
                    else self.dataset.total_rollout_timesteps
                ), (
                    f"Event end index {event_end_idx} exceeds dataset length {self.dataset.total_rollout_timesteps} for event_idx {event_idx}."
                )
            event_dataset = self.dataset[event_start_idx:event_end_idx]
            dataloader = DataLoader(event_dataset, batch_size=1, shuffle=False)
            actual_event_id = self.dataset.hec_ras_run_ids[event_idx]

            # 2D sliding window
            sliding_window = (
                self.dataset[event_start_idx]
                .x[:, self.start_node_target_idx : self.end_node_target_idx]
                .clone()
            )
            sliding_window = sliding_window.to(self.device)
            edge_sliding_window = (
                self.dataset[event_start_idx]
                .edge_attr[:, self.start_edge_target_idx : self.end_edge_target_idx]
                .clone()
            )
            edge_sliding_window = edge_sliding_window.to(self.device)

            # 1D sliding window
            sliding_window_1d = (
                self.dataset[event_start_idx]
                .x_1d[:, self.start_1d_node_target_idx : self.end_1d_node_target_idx]
                .clone()
            )
            sliding_window_1d = sliding_window_1d.to(self.device)
            edge_sliding_window_1d = (
                self.dataset[event_start_idx]
                .edge_attr_1d[:, self.start_1d_edge_target_idx : self.end_1d_edge_target_idx]
                .clone()
            )
            edge_sliding_window_1d = edge_sliding_window_1d.to(self.device)


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

                # 2D features
                x = torch.concat(
                    [
                        graph.x[:, : self.start_node_target_idx],
                        sliding_window,
                        graph.x[:, self.end_node_target_idx :],
                    ],
                    dim=1,
                )
                edge_index, edge_attr = graph.edge_index, graph.edge_attr
                edge_attr = torch.concat(
                    [
                        graph.edge_attr[:, : self.start_edge_target_idx],
                        edge_sliding_window,
                        graph.edge_attr[:, self.end_edge_target_idx :],
                    ],
                    dim=1,
                )

                # 1D features
                x_1d = torch.concat(
                    [
                        graph.x_1d[:, : self.start_1d_node_target_idx],
                        sliding_window_1d,
                        graph.x_1d[:, self.end_1d_node_target_idx :],
                    ],
                    dim=1,
                )
                edge_index_1d, edge_attr_1d = graph.edge_index_1d, graph.edge_attr_1d
                edge_attr_1d = torch.concat(
                    [
                        graph.edge_attr_1d[:, : self.start_1d_edge_target_idx],
                        edge_sliding_window_1d,
                        graph.edge_attr_1d[:, self.end_1d_edge_target_idx :],
                    ],
                    dim=1,
                )

                if self.feature_alignment == "inject_rainfall":
                     # print("Selected inject_rainfall feature alignment")  # --- IGNORE ---
                    _, x_1d, edge_attr, edge_attr_1d = self.feature_aligner.inject_nearest_rainfall_to_1d(
                        x, x_1d, edge_attr, edge_attr_1d
                    )
                elif self.feature_alignment == "common_no_rainfall_1d":
                    # print("Selected common feature no rainfall 1d alignment")  # --- IGNORE ---
                    x, x_1d, edge_attr, edge_attr_1d = self.feature_aligner.align_common_features_no_rainfall_1d(
                        x, x_1d, edge_attr, edge_attr_1d
                    )
                elif self.feature_alignment == "common":
                    # print("Selected common feature alignment")  # --- IGNORE ---
                    x, x_1d, edge_attr, edge_attr_1d = self.feature_aligner.align_common_features(
                        x, x_1d, edge_attr, edge_attr_1d
                    )

                # Model prediction - returns both 2D and 1D node predictions
                pred_diff, pred_diff_1d = self.model(
                    x, edge_index, edge_attr, x_1d, edge_index_1d, edge_attr_1d, graph.edge_index_1d_2d
                )

                # Override boundary conditions for 2D predictions
                pred_diff[self.boundary_nodes_mask] = graph.y[self.boundary_nodes_mask]

                # Override boundary conditions for 1D predictions (if applicable)
                if (
                    hasattr(self, "boundary_nodes_1d_mask")
                    and self.boundary_nodes_1d_mask is not None
                ):
                    pred_diff_1d[self.boundary_nodes_1d_mask] = graph.y_1d[
                        self.boundary_nodes_1d_mask
                    ]

                # 2D predictions
                prev_node_pred = sliding_window[:, [-1]]
                pred = prev_node_pred + pred_diff

                # 1D predictions
                prev_node_pred_1d = sliding_window_1d[:, [-1]]
                pred_1d = prev_node_pred_1d + pred_diff_1d

                # Physics-informed loss (for 2D - if applicable)
                if self.include_physics_loss:
                    prev_edge_pred = physics_utils.get_curr_flow_from_edge_features(
                        edge_attr, self.dataset.previous_timesteps
                    )
                    validation_stats.update_physics_informed_stats_for_timestep(
                        pred,
                        prev_node_pred,
                        prev_edge_pred,
                        graph,
                        TEST_LOCAL_MASS_LOSS_NODES,
                    )

                # Update sliding windows
                sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)
                sliding_window_1d = torch.concat(
                    (sliding_window_1d[:, 1:], pred_1d), dim=1
                )

                # ===== 2D Node Validation =====
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

                # Save 2D predictions
                if save_predictions:
                    pred_2d_cpu = pred.cpu().numpy().flatten()
                    # Get 1D node IDs
                    node_ids_2d = (
                        graph.node_id_2d.cpu().numpy()
                        if hasattr(graph, "node_id_2d")
                        else range(len(pred_2d_cpu))
                    )

                    for node_idx, (node_id, water_level) in enumerate(
                        zip(node_ids_2d, pred_2d_cpu)
                    ):
                        predictions_list.append(
                            {
                                "row_id": row_id,
                                "model_id": model_id,
                                "event_id": actual_event_id,
                                "node_type": 2,  # 2D nodes
                                "node_id": node_id,
                                "water_level": float(water_level),
                            }
                        )
                        row_id += 1

                pred = pred[self.non_boundary_nodes_mask]
                label = label[self.non_boundary_nodes_mask]

                validation_stats.update_stats_for_timestep(
                    pred.cpu(),
                    label.cpu(),
                    pred_ground_elevation=self.ground_elevation_2d[
                        self.non_boundary_nodes_mask
                    ].cpu(),
                    label_ground_elevation=self.ground_elevation_2d[
                        self.non_boundary_nodes_mask
                    ].cpu(),
                    water_threshold=self.threshold_per_cell_2d[
                        self.non_boundary_nodes_mask
                    ],
                    timestamp=graph.timestep if hasattr(graph, "timestep") else None,
                )

                # ===== 1D Node Validation =====
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

                # Save 1D predictions
                if save_predictions:
                    pred_1d_cpu = pred_1d.cpu().numpy().flatten()
                    # Get 1D node IDs
                    node_ids_1d = (
                        graph.node_id_1d.cpu().numpy()
                        if hasattr(graph, "node_id_1d")
                        else range(len(pred_1d_cpu))
                    )

                    for node_idx, (node_id, water_level) in enumerate(
                        zip(node_ids_1d, pred_1d_cpu)
                    ):
                        predictions_list.append(
                            {
                                "row_id": row_id,
                                "model_id": model_id,
                                "event_id": actual_event_id,
                                "node_type": 1,  # 1D nodes
                                "node_id": node_id,
                                "water_level": float(water_level),
                            }
                        )
                        row_id += 1

                validation_stats.update_1d_stats_for_timestep(
                    pred_1d.cpu(),
                    label_1d.cpu(),
                    pred_ground_elevation=self.surface_elevation_1d.cpu(),
                    label_ground_elevation=self.surface_elevation_1d.cpu(),
                    water_threshold=self.threshold_per_cell_1d,
                    timestamp=graph.timestep if hasattr(graph, "timestep") else None,
                )

        validation_stats.end_validate()
        # Save predictions to CSV
        if save_predictions:
            predictions_dir = Path(predictions_dir)
            predictions_dir.mkdir(exist_ok=True, parents=True)
            
            df = pd.DataFrame(predictions_list)
            output_file = predictions_dir / f"predictions_event_{actual_event_id}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved predictions to {output_file}")
            
            return df
