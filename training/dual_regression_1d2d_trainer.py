import os

from contextlib import redirect_stdout
import torch
from torch import Tensor
from testing import DualAutoregressive1D2DTester
from typing import Callable
from utils import LossScaler, physics_utils, train_utils

from .node_regression_1d2d_trainer import NodeRegression1D2DTrainer
from .edge_regression_1d2d_trainer import EdgeRegression1D2DTrainer


class DualRegression1D2DTrainer(NodeRegression1D2DTrainer, EdgeRegression1D2DTrainer):
    def __init__(
        self,
        edge_loss_func: Callable,
        edge_pred_loss_scale: float = 1.0,
        edge_loss_weight: float = 1.0,
        edge_1d_pred_loss_scale: float = 1.0,
        edge_1d_loss_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        ds = self.dataloader.dataset
        self.boundary_edges_mask = ds.boundary_condition.boundary_edges_mask

        # 2D Edge (existing)
        self.edge_loss_func = edge_loss_func
        self.edge_loss_weight = edge_loss_weight
        self.edge_loss_scaler = LossScaler(initial_scale=edge_pred_loss_scale)

        # 1D Edge
        self.edge_1d_loss_func = edge_loss_func
        self.edge_1d_loss_weight = edge_1d_loss_weight
        self.edge_1d_loss_scaler = LossScaler(initial_scale=edge_1d_pred_loss_scale)

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()

            running_pred_loss_2d = 0.0
            running_edge_pred_loss_2d = 0.0
            running_pred_loss_1d = 0.0
            running_edge_pred_loss_1d = 0.0
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

                # 2D data
                x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
                # 1D data
                x_1d, edge_index_1d, edge_attr_1d = (
                    batch.x_1d,
                    batch.edge_index_1d,
                    batch.edge_attr_1d,
                )

                # Model forward pass (returns 4 predictions)
                pred_diff_2d, edge_pred_diff_2d, pred_diff_1d, edge_pred_diff_1d = (
                    self.model(
                        x,
                        edge_index,
                        edge_attr,
                        x_1d,
                        edge_index_1d,
                        edge_attr_1d,
                        batch.edge_index_1d_2d,
                    )
                )

                # Apply boundary conditions to only 2D edges
                pred_diff_2d, edge_pred_diff_2d = self._override_pred_bc(pred_diff_2d, edge_pred_diff_2d, batch)

                # 2D Node loss
                pred_loss_2d = self._compute_node_loss(pred_diff_2d, batch.y)
                pred_loss_2d = self._scale_node_pred_loss(epoch, pred_loss_2d)
                running_pred_loss_2d += pred_loss_2d.item()

                # 2D Edge loss
                edge_pred_loss_2d = self._compute_edge_loss(edge_pred_diff_2d, batch.y_edge)
                edge_pred_loss_2d = self._scale_edge_pred_loss(
                    epoch, pred_loss_2d, edge_pred_loss_2d
                )
                running_edge_pred_loss_2d += edge_pred_loss_2d.item()

                # 1D Node loss
                pred_loss_1d = self._compute_node_loss(pred_diff_1d, batch.y_1d)
                running_pred_loss_1d += pred_loss_1d.item()

                # 1D Edge loss
                edge_pred_loss_1d = self._compute_edge_loss(edge_pred_diff_1d, batch.y_1d_edge)
                edge_pred_loss_1d = self._scale_edge_pred_loss(
                    epoch, pred_loss_1d, edge_pred_loss_1d
                )
                running_edge_pred_loss_1d += edge_pred_loss_1d.item()

                loss = (
                    pred_loss_2d + edge_pred_loss_2d + pred_loss_1d + edge_pred_loss_1d
                )

                # Physics loss (if applicable - using 2D data only)
                if self.use_physics_loss:
                    previous_timesteps = self.dataloader.dataset.previous_timesteps
                    curr_water_volume, curr_face_flow = (
                        physics_utils.get_physics_info_node_edge(
                            x, edge_attr, previous_timesteps, batch
                        )
                    )
                    
                    area_idx = self.dataloader.dataset.STATIC_NODE_FEATURES.index('area')
                    node_areas = x[:, area_idx:area_idx+1]
                    
                    # Convert water level change to volume change
                    pred_volume_diff = pred_diff_2d * node_areas
                    pred_volume = curr_water_volume + pred_volume_diff

                    global_loss, local_loss = self._get_physics_loss(
                        epoch,
                        pred_volume,
                        curr_water_volume,
                        curr_face_flow,
                        pred_loss_2d,
                        batch,
                    )
                    running_global_mass_loss += global_loss.item()
                    running_local_mass_loss += local_loss.item()
                    loss = loss + global_loss + local_loss

                loss.backward()
                self.optimizer.step()

            running_loss = (running_pred_loss_2d + running_edge_pred_loss_2d + 
                          running_pred_loss_1d + running_edge_pred_loss_1d + 
                          running_global_mass_loss + running_local_mass_loss)
            running_losses = (running_loss, running_pred_loss_2d, running_edge_pred_loss_2d, 
                            running_pred_loss_1d, running_edge_pred_loss_1d,
                            running_global_mass_loss, running_local_mass_loss)
            epoch_losses = train_utils.divide_losses(running_losses, len(self.dataloader))
            (epoch_loss, pred_epoch_loss_2d, edge_pred_epoch_loss_2d, 
             pred_epoch_loss_1d, edge_pred_epoch_loss_1d, 
             global_mass_epoch_loss, local_mass_epoch_loss) = epoch_losses

            # Logging
            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tTotal Loss: {epoch_loss:.4e}\n'
            logging_str += f'\t2D Node Prediction Loss: {pred_epoch_loss_2d:.4e}\n'
            logging_str += f'\t2D Edge Prediction Loss: {edge_pred_epoch_loss_2d:.4e}\n'
            logging_str += f'\t1D Node Prediction Loss: {pred_epoch_loss_1d:.4e}\n'
            logging_str += f'\t1D Edge Prediction Loss: {edge_pred_epoch_loss_1d:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('2d_prediction_loss', pred_epoch_loss_2d)
            self.training_stats.add_loss_component('2d_edge_prediction_loss', edge_pred_epoch_loss_2d)
            self.training_stats.add_loss_component('1d_prediction_loss', pred_epoch_loss_1d)
            self.training_stats.add_loss_component('1d_edge_prediction_loss', edge_pred_epoch_loss_1d)

            if self.use_physics_loss:
                self._log_epoch_physics_loss(global_mass_epoch_loss, local_mass_epoch_loss)

            self._update_loss_scaler_for_epoch(epoch)

            if hasattr(self, 'early_stopping'):
                val_node_rmse, val_edge_rmse, val_node_1d_rmse, val_edge_1d_rmse = self.validate()
                self.training_stats.log(f'\n\tValidation 2D Node RMSE: {val_node_rmse:.4e}')
                self.training_stats.log(f'\tValidation 2D Edge RMSE: {val_edge_rmse:.4e}')
                self.training_stats.log(f'\tValidation 1D Node RMSE: {val_node_1d_rmse:.4e}')
                self.training_stats.log(f'\tValidation 1D Edge RMSE: {val_edge_1d_rmse:.4e}')
                self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)
                self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)
                self.training_stats.add_val_loss_component('val_1d_node_rmse', val_node_1d_rmse)
                self.training_stats.add_val_loss_component('val_1d_edge_rmse', val_edge_1d_rmse)

                # Early stopping on combined metric
                combined_metric = val_node_rmse + val_edge_rmse + val_node_1d_rmse + val_edge_1d_rmse
                if self.early_stopping(combined_metric, self.model):
                    self.training_stats.log(f'Early stopping triggered at epoch {epoch + 1}.')
                    break

        self.training_stats.end_train()
        self.training_stats.add_additional_info('1d_edge_scaled_loss_ratios', self.edge_1d_loss_scaler.scaled_loss_ratio_history)
        self.training_stats.add_additional_info('2d_edge_scaled_loss_ratios', self.edge_loss_scaler.scaled_loss_ratio_history)
        self._add_scaled_physics_loss_history()

    def validate(self):
        val_tester = DualAutoregressive1D2DTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device,
        )

        # Check model parameters for nan BEFORE validation
        print("\n=== Checking Model Parameters ===")
        has_nan_params = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"NAN in parameter: {name}")
                has_nan_params = True
        
        if has_nan_params:
            print("ERROR: Model has NaN parameters!")
            return float('nan'), float('nan'), float('nan'), float('nan')
        else:
            print("Model parameters OK (no NaN)")

        # with open(os.devnull, "w") as f, redirect_stdout(f):
        val_tester.test()

        node_rmse = val_tester.get_avg_node_rmse()
        edge_rmse = val_tester.get_avg_edge_rmse()
        node_1d_rmse = val_tester.get_avg_node_1d_rmse()
        edge_1d_rmse = val_tester.get_avg_edge_1d_rmse()

        print(f"Validation results: node_rmse={node_rmse}, edge_rmse={edge_rmse}, "
          f"node_1d_rmse={node_1d_rmse}, edge_1d_rmse={edge_1d_rmse}")

        return node_rmse, edge_rmse, node_1d_rmse, edge_1d_rmse

    def _override_pred_bc(self, pred: Tensor, edge_pred: Tensor, batch) -> Tensor:
        pred = NodeRegression1D2DTrainer._override_pred_bc(self, pred, batch)
        edge_pred = EdgeRegression1D2DTrainer._override_pred_bc(self, edge_pred, batch)
        return pred, edge_pred

    # ========= Methods for scaling losses =========

    def _scale_edge_pred_loss(
        self, epoch: int, basis_loss: Tensor, edge_pred_loss: Tensor
    ) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.add_epoch_loss_ratio(basis_loss, edge_pred_loss)
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss)
        else:
            scaled_edge_pred_loss = (
                self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_loss_weight
            )
        return scaled_edge_pred_loss

    def _update_loss_scaler_for_epoch(self, epoch: int):
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.update_scale_from_epoch()
            self.training_stats.log(
                f"\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}"
            )
        NodeRegression1D2DTrainer._update_loss_scaler_for_epoch(self, epoch)
