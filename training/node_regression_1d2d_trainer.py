import os
import numpy as np

from contextlib import redirect_stdout
from data import FloodEvent1D2DDataset
from torch import Tensor
from testing import NodeAutoregressive1D2DTester
from utils import physics_utils, train_utils

from .physics_informed_1d2d_trainer import PhysicsInformed1D2DTrainer


class NodeRegression1D2DTrainer(PhysicsInformed1D2DTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: FloodEvent1D2DDataset = self.dataloader.dataset
        self.boundary_nodes_mask = ds.boundary_condition.boundary_nodes_mask

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()

            running_pred_loss_2d = 0.0
            running_pred_loss_1d = 0.0
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
                x_1d, edge_index_1d = (
                    batch.x_1d,
                    batch.edge_index_1d,
                )

                # Model forward pass - returns both 2D and 1D node predictions
                pred_diff_2d, pred_diff_1d = self.model(
                    x,
                    edge_index,
                    x_1d,
                    edge_index_1d,
                    batch.edge_index_1d_2d,
                )

                # Override boundary conditions
                pred_diff_2d = self._override_pred_bc(pred_diff_2d, batch)

                # pred_diff_1d = self._override_pred_bc_1d(pred_diff_1d, batch) # 1D Nodes have no boundary condition

                # Compute 2D node loss
                loss_2d = self._compute_node_loss(pred_diff_2d, batch.y)
                loss_2d = self._scale_node_pred_loss(epoch, loss_2d)
                running_pred_loss_2d += loss_2d.item()

                # Compute 1D node loss (same loss function)
                loss_1d = self._compute_node_loss(pred_diff_1d, batch.y_1d)
                running_pred_loss_1d += loss_1d.item()

                # Total prediction loss
                loss = loss_2d + loss_1d

                # Physics-informed loss (if applicable, for 2D)
                if self.use_physics_loss:
                    previous_timesteps = self.dataloader.dataset.previous_timesteps
                    curr_water_volume, curr_face_flow = (
                        physics_utils.get_physics_info_node_edge(
                            x, edge_attr, previous_timesteps, batch
                        )
                    )
                    pred = curr_water_volume + pred_diff_2d
                    global_loss, local_loss = self._get_physics_loss(
                        epoch, pred, curr_water_volume, curr_face_flow, loss_2d, batch
                    )
                    running_global_mass_loss += global_loss.item()
                    running_local_mass_loss += local_loss.item()
                    loss = loss + global_loss + local_loss

                loss.backward()
                self.optimizer.step()

            running_loss = (
                running_pred_loss_2d
                + running_pred_loss_1d
                + running_global_mass_loss
                + running_local_mass_loss
            )
            running_losses = (
                running_loss,
                running_pred_loss_2d,
                running_pred_loss_1d,
                running_global_mass_loss,
                running_local_mass_loss,
            )
            epoch_losses = train_utils.divide_losses(
                running_losses, len(self.dataloader)
            )
            (
                epoch_loss,
                pred_epoch_loss_2d,
                pred_epoch_loss_1d,
                global_mass_epoch_loss,
                local_mass_epoch_loss,
            ) = epoch_losses

            logging_str = f"Epoch [{epoch + 1}/{self.num_epochs}]\n"
            logging_str += f"\tTotal Loss: {epoch_loss:.4e}\n"
            logging_str += f"\t2D Node Prediction Loss: {pred_epoch_loss_2d:.4e}\n"
            logging_str += f"\t1D Node Prediction Loss: {pred_epoch_loss_1d:.4e}"
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component(
                "prediction_loss_2d", pred_epoch_loss_2d
            )
            self.training_stats.add_loss_component(
                "prediction_loss_1d", pred_epoch_loss_1d
            )

            if self.use_physics_loss:
                self._log_epoch_physics_loss(
                    global_mass_epoch_loss, local_mass_epoch_loss
                )

            self._update_loss_scaler_for_epoch(epoch)

            if hasattr(self, "early_stopping"):
                val_node_rmse_2d, val_node_rmse_1d = self.validate()
                self.training_stats.log(
                    f"\n\tValidation 2D Node RMSE: {val_node_rmse_2d:.4e}"
                )
                self.training_stats.log(
                    f"\tValidation 1D Node RMSE: {val_node_rmse_1d:.4e}"
                )
                self.training_stats.add_val_loss_component(
                    "val_node_rmse_2d", val_node_rmse_2d
                )
                self.training_stats.add_val_loss_component(
                    "val_node_rmse_1d", val_node_rmse_1d
                )

                # Early stopping based on combined metric
                combined_metric = val_node_rmse_2d + val_node_rmse_1d
                if self.early_stopping(combined_metric, self.model):
                    self.training_stats.log(
                        f"Early stopping triggered at epoch {epoch + 1}."
                    )
                    break

        self.training_stats.end_train()
        self._add_scaled_physics_loss_history()

    def validate(self):
        val_tester = NodeAutoregressive1D2DTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device,
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        node_rmse_2d = val_tester.get_avg_node_rmse()
        node_rmse_1d = val_tester.get_avg_node_1d_rmse()
        return node_rmse_2d, node_rmse_1d

    def _compute_node_loss(self, pred: Tensor, label: Tensor) -> Tensor:
        return self.loss_func(pred, label)

    def _override_pred_bc(self, pred: Tensor, batch) -> Tensor:
        batch_boundary_nodes_mask = np.tile(self.boundary_nodes_mask, batch.num_graphs)
        pred[batch_boundary_nodes_mask] = batch.y[batch_boundary_nodes_mask].float()
        return pred
