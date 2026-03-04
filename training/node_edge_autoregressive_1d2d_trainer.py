import os
import time
import numpy as np
from testing import NodeEdgeAutoregressive1D2DTester
import torch

from contextlib import redirect_stdout
from torch import Tensor
from data import AutoregressiveFlood1D2DDataset
from typing import Tuple
from utils import EarlyStopping, physics_utils, train_utils

from .base_autoregressive_1d2d_trainer import BaseAutoregressive1D2DTrainer
from .physics_informed_1d2d_trainer import PhysicsInformed1D2DTrainer
from data.feature_aligner import BatchTensorAligner

class NodeEdgeAutoregressive1D2DTrainer(
    BaseAutoregressive1D2DTrainer, PhysicsInformed1D2DTrainer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: AutoregressiveFlood1D2DDataset = self.dataloader.dataset
        # Get non-boundary nodes/edges and threshold for metric computation
        self.boundary_nodes_mask = ds.boundary_condition.boundary_nodes_mask

        # Get sliding window indices
        sliding_window_length = ds.previous_timesteps + 1
        target_nodes_idx = ds.DYNAMIC_NODE_FEATURES.index(ds.NODE_TARGET_FEATURE)
        target_edges_idx = ds.DYNAMIC_EDGE_FEATURES.index(ds.EDGE_TARGET_FEATURE)
        self.start_node_target_idx = ds.num_static_node_features + (
            target_nodes_idx * sliding_window_length
        )
        self.end_node_target_idx = self.start_node_target_idx + sliding_window_length
        self.start_edge_target_idx = ds.num_static_edge_features + (
            target_edges_idx * sliding_window_length
        )
        self.end_edge_target_idx = self.start_edge_target_idx + sliding_window_length

        target_1d_nodes_idx = ds.DYNAMIC_1D_NODE_FEATURES.index(
            ds.NODE_1D_TARGET_FEATURE
        )
        target_1d_edges_idx = ds.DYNAMIC_1D_EDGE_FEATURES.index(
            ds.EDGE_1D_TARGET_FEATURE
        )
        self.start_1d_node_target_idx = ds.num_static_1d_node_features + (
            target_1d_nodes_idx * sliding_window_length
        )
        self.end_1d_node_target_idx = (
            self.start_1d_node_target_idx + sliding_window_length
        )
        self.start_1d_edge_target_idx = ds.num_static_1d_edge_features + (
            target_1d_edges_idx * sliding_window_length
        )
        self.end_1d_edge_target_idx = (
            self.start_1d_edge_target_idx + sliding_window_length
        )
        self.feature_aligner = BatchTensorAligner(ds).to(self.device)

    def train(self):
        """Multi-step-ahead loss with curriculum learning."""
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps
        current_timestep_epochs = 0

        for epoch in range(self.num_epochs):
            train_start_time = time.time()

            train_losses = self._train_model(epoch, current_num_timesteps)
            (
                epoch_loss,
                pred_epoch_loss,
                pred_1d_epoch_loss,
                global_mass_epoch_loss,
                local_mass_epoch_loss,
            ) = train_losses

            logging_str = f"Epoch [{epoch + 1}/{self.num_epochs}]\n"
            logging_str += f"\tLoss: {epoch_loss:.4e}\n"
            logging_str += f"\t2D Node Prediction Loss: {pred_epoch_loss:.4e}\n"
            logging_str += f"\t1D Node Prediction Loss: {pred_1d_epoch_loss:.4e}"
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component(
                "prediction_loss_2d", pred_epoch_loss
            )
            self.training_stats.add_loss_component(
                "prediction_loss_1d", pred_1d_epoch_loss
            )

            if self.use_physics_loss:
                self._log_epoch_physics_loss(
                    global_mass_epoch_loss, local_mass_epoch_loss
                )

            self._update_loss_scaler_for_epoch(epoch)

            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            self.training_stats.log(
                f"\tEpoch Train Duration: {train_duration:.2f} seconds"
            )

            val_node_rmse, val_1d_node_rmse = self.validate()
            self.training_stats.log(f"\n\tValidation 2D Node RMSE: {val_node_rmse:.4e}")
            self.training_stats.log(
                f"\tValidation 1D Node RMSE: {val_1d_node_rmse:.4e}"
            )
            self.training_stats.add_val_loss_component(
                "val_node_rmse_2d", val_node_rmse
            )
            self.training_stats.add_val_loss_component(
                "val_node_rmse_1d", val_1d_node_rmse
            )

            current_timestep_epochs += 1

            combined_rmse = (val_node_rmse + val_1d_node_rmse) / 2
            is_early_stopped = self.early_stopping(combined_rmse, self.model)
            is_max_exceeded = (
                self.max_curriculum_epochs is not None
                and current_timestep_epochs >= self.max_curriculum_epochs
            )

            if is_early_stopped or is_max_exceeded:
                if current_num_timesteps < self.total_num_timesteps:
                    self.training_stats.log(
                        f"\tCurriculum learning for {current_num_timesteps} steps ended after {current_timestep_epochs} epochs."
                    )
                    current_num_timesteps += self.timestep_increment
                    current_timestep_epochs = 0
                    self.early_stopping = EarlyStopping(
                        patience=self.early_stopping.patience
                    )
                    self.training_stats.log(
                        f"\tIncreased current_num_timesteps to {current_num_timesteps} timesteps."
                    )
                    self.lr_scheduler.step()
                    self.training_stats.log(
                        f"\tDecayed learning rate to {self.lr_scheduler.get_last_lr()[0]:.4e}."
                    )
                    continue

                self.training_stats.log("Training completed due to early stopping.")
                break

        self.training_stats.end_train()
        self._add_scaled_physics_loss_history()

    def _train_model(
        self, epoch: int, current_num_timesteps: int
    ) -> Tuple[float, float]:
        self.model.train()

        running_pred_loss = 0.0
        running_pred_1d_loss = 0.0
        running_global_mass_loss = 0.0
        running_local_mass_loss = 0.0

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            batch.x = batch.x.float()
            batch.edge_attr = batch.edge_attr.float()
            batch.y = batch.y.float()
            batch.y_edge = batch.y_edge.float()
            batch.x_1d = batch.x_1d.float()
            batch.edge_attr_1d = batch.edge_attr_1d.float()
            batch.y_1d = batch.y_1d.float()
            batch.y_1d_edge = batch.y_1d_edge.float()

            x, edge_index, edge_attr = (
                batch.x[:, :, 0],
                batch.edge_index,
                batch.edge_attr[:, :, 0],
            )

            x_1d, edge_index_1d, edge_attr_1d = (
                batch.x_1d[:, :, 0],
                batch.edge_index_1d,
                batch.edge_attr_1d[:, :, 0],
            )

            edge_index_1d_2d = batch.edge_index_1d_2d

            total_batch_loss = 0.0
            total_batch_pred_loss = 0.0
            total_batch_pred_1d_loss = 0.0
            total_batch_global_mass_loss = 0.0
            total_batch_local_mass_loss = 0.0

            sliding_window = x[
                :, self.start_node_target_idx : self.end_node_target_idx
            ].clone()
            sliding_window_1d = x_1d[
                :, self.start_1d_node_target_idx : self.end_1d_node_target_idx
            ].clone()

            edge_sliding_window = edge_attr[
                :, self.start_edge_target_idx : self.end_edge_target_idx
            ].clone()
            edge_sliding_window_1d = edge_attr_1d[
                :, self.start_1d_edge_target_idx : self.end_1d_edge_target_idx
            ].clone()

            for i in range(current_num_timesteps):
                x = batch.x[:, :, i]
                edge_attr = batch.edge_attr[:, :, i]
                x_1d = batch.x_1d[:, :, i]
                edge_attr_1d = batch.edge_attr_1d[:, :, i]

                # Override graph data with sliding window
                x = torch.concat(
                    [
                        x[:, : self.start_node_target_idx],
                        sliding_window,
                        x[:, self.end_node_target_idx :],
                    ],
                    dim=1,
                )

                edge_attr = torch.concat(
                    [
                        edge_attr[:, : self.start_edge_target_idx],
                        edge_sliding_window,
                        edge_attr[:, self.end_edge_target_idx :],
                    ],
                    dim=1,
                )

                x_1d = torch.concat(
                    [
                        x_1d[:, : self.start_1d_node_target_idx],
                        sliding_window_1d,
                        x_1d[:, self.end_1d_node_target_idx :],
                    ],
                    dim=1,
                )

                edge_attr_1d = torch.concat(
                    [
                        edge_attr_1d[:, : self.start_1d_edge_target_idx],
                        edge_sliding_window_1d,
                        edge_attr_1d[:, self.end_1d_edge_target_idx :],
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

                pred_diff, pred_diff_1d = self.model(
                    x,
                    edge_index,
                    edge_attr,
                    x_1d,
                    edge_index_1d,
                    edge_attr_1d,
                    edge_index_1d_2d,
                )
                # Override boundary conditions
                pred_diff = self._override_pred_bc(pred_diff, batch, i)

                pred_loss = self._compute_node_loss(pred_diff, batch, i)
                pred_loss = self._scale_node_pred_loss(epoch, pred_loss)
                total_batch_pred_loss += pred_loss.item()

                pred_1d_loss = self._compute_node_1d_loss(pred_diff_1d, batch, i)
                pred_1d_loss = self._scale_node_pred_loss(
                    epoch, pred_1d_loss
                )  # Reuse same scaling
                total_batch_pred_1d_loss += pred_1d_loss.item()

                step_loss = pred_loss + pred_1d_loss

                previous_timesteps = self.dataloader.dataset.previous_timesteps
                prev_node_pred = sliding_window[:, [-1]]
                pred = prev_node_pred + pred_diff

                prev_node_pred_1d = sliding_window_1d[:, [-1]]
                pred_1d = prev_node_pred_1d + pred_diff_1d

                if self.use_physics_loss:
                    curr_face_flow = physics_utils.get_curr_flow_from_edge_features(
                        edge_attr, previous_timesteps
                    )
                    global_loss, local_loss = self._get_physics_loss(
                        epoch,
                        pred,
                        prev_node_pred,
                        curr_face_flow,
                        pred_loss,
                        batch,
                        current_timestep=i,
                    )
                    total_batch_global_mass_loss += global_loss.item()
                    total_batch_local_mass_loss += local_loss.item()
                    step_loss = step_loss + global_loss + local_loss

                total_batch_loss = total_batch_loss + step_loss

                if i < current_num_timesteps - 1:  # Don't update on last iteration
                    next_sliding_window = torch.cat(
                        (sliding_window[:, 1:], pred), dim=1
                    )
                    sliding_window = next_sliding_window
                    next_sliding_window_1d = torch.cat(
                        (sliding_window_1d[:, 1:], pred_1d), dim=1
                    )
                    sliding_window_1d = next_sliding_window_1d

            avg_batch_loss = total_batch_loss / current_num_timesteps
            avg_batch_loss.backward()
            self._clip_gradients()
            self.optimizer.step()

            # Loss Updates
            total_losses = (
                total_batch_pred_loss,
                total_batch_pred_1d_loss,
                total_batch_global_mass_loss,
                total_batch_local_mass_loss,
            )
            avg_losses = train_utils.divide_losses(total_losses, current_num_timesteps)
            (
                avg_pred_loss,
                avg_pred_1d_loss,
                avg_global_mass_loss,
                avg_local_mass_loss,
            ) = avg_losses

            running_pred_loss += avg_pred_loss
            running_pred_1d_loss += avg_pred_1d_loss
            running_global_mass_loss += avg_global_mass_loss
            running_local_mass_loss += avg_local_mass_loss

        running_loss = (
            running_pred_loss
            + running_pred_1d_loss
            + running_global_mass_loss
            + running_local_mass_loss
        )
        running_losses = (
            running_loss,
            running_pred_loss,
            running_pred_1d_loss,
            running_global_mass_loss,
            running_local_mass_loss,
        )
        epoch_losses = train_utils.divide_losses(running_losses, len(self.dataloader))
        (
            epoch_loss,
            pred_epoch_loss,
            pred_1d_epoch_loss,
            global_mass_epoch_loss,
            local_mass_epoch_loss,
        ) = epoch_losses

        return (
            epoch_loss,
            pred_epoch_loss,
            pred_1d_epoch_loss,
            global_mass_epoch_loss,
            local_mass_epoch_loss,
        )

    def validate(self):
        val_tester = NodeEdgeAutoregressive1D2DTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device,
            feature_alignment=self.feature_alignment,
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        node_rmse = val_tester.get_avg_node_rmse()
        node_1d_rmse = val_tester.get_avg_node_1d_rmse()
        return node_rmse, node_1d_rmse

    def _compute_node_loss(self, pred: Tensor, batch, timestep: int) -> Tensor:
        label = batch.y[:, :, timestep]
        return self.loss_func(pred, label)

    def _compute_node_1d_loss(self, pred: Tensor, batch, timestep: int) -> Tensor:
        label = batch.y_1d[:, :, timestep]
        return self.loss_func(pred, label)

    def _override_pred_bc(self, pred: Tensor, batch, timestep: int) -> Tensor:
        batch_boundary_nodes_mask = np.tile(self.boundary_nodes_mask, batch.num_graphs)
        pred[batch_boundary_nodes_mask] = batch.y[
            batch_boundary_nodes_mask, :, timestep
        ]
        return pred
