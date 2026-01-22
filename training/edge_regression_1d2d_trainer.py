import os
import numpy as np

from contextlib import redirect_stdout
from data import FloodEvent1D2DDataset
from testing import EdgeAutoregressive1D2DTester
from torch import Tensor

from .base_1d2d_trainer import Base1D2DTrainer


class EdgeRegression1D2DTrainer(Base1D2DTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: FloodEvent1D2DDataset = self.dataloader.dataset
        self.boundary_edges_mask = ds.boundary_condition.boundary_edges_mask

    def train(self):
        self.training_stats.start_train()
        for epoch in range(self.num_epochs):
            self.model.train()
            running_edge_pred_loss_2d = 0.0
            running_edge_pred_loss_1d = 0.0

            for batch in self.dataloader:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)

                # 2D data
                x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
                # 1D data
                x_1d, edge_index_1d, edge_attr_1d = (
                    batch.x_1d,
                    batch.edge_index_1d,
                    batch.edge_attr_1d,
                )

                # Model forward pass
                edge_pred_2d, edge_pred_1d = self.model(
                    x,
                    edge_index,
                    edge_attr,
                    x_1d,
                    edge_index_1d,
                    edge_attr_1d,
                    batch.edge_index_1d_2d,
                )

                # Override boundary conditions for 2D only
                edge_pred_2d = self._override_pred_bc(edge_pred_2d, batch)

                # Compute losses (same loss function for both)
                loss_2d = self._compute_edge_loss(edge_pred_2d, batch.y_edge)
                loss_1d = self._compute_edge_loss(edge_pred_1d, batch.y_1d_edge)

                running_edge_pred_loss_2d += loss_2d.item()
                running_edge_pred_loss_1d += loss_1d.item()

                # Total loss
                loss = loss_2d + loss_1d

                loss.backward()
                self.optimizer.step()

            edge_pred_epoch_loss_2d = running_edge_pred_loss_2d / len(self.dataloader)
            edge_pred_epoch_loss_1d = running_edge_pred_loss_1d / len(self.dataloader)
            total_epoch_loss = edge_pred_epoch_loss_2d + edge_pred_epoch_loss_1d

            logging_str = f"Epoch [{epoch + 1}/{self.num_epochs}]\n"
            logging_str += f"\tTotal Loss: {total_epoch_loss:.4e}\n"
            logging_str += f"\t2D Edge Prediction Loss: {edge_pred_epoch_loss_2d:.4e}\n"
            logging_str += f"\t1D Edge Prediction Loss: {edge_pred_epoch_loss_1d:.4e}"
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(total_epoch_loss)
            self.training_stats.add_loss_component(
                "edge_prediction_loss_2d", edge_pred_epoch_loss_2d
            )
            self.training_stats.add_loss_component(
                "edge_prediction_loss_1d", edge_pred_epoch_loss_1d
            )

            if hasattr(self, "early_stopping"):
                val_edge_rmse_2d, val_edge_rmse_1d = self.validate()
                self.training_stats.log(
                    f"\n\tValidation 2D Edge RMSE: {val_edge_rmse_2d:.4e}"
                )
                self.training_stats.log(
                    f"\tValidation 1D Edge RMSE: {val_edge_rmse_1d:.4e}"
                )
                self.training_stats.add_val_loss_component(
                    "val_edge_rmse_2d", val_edge_rmse_2d
                )
                self.training_stats.add_val_loss_component(
                    "val_edge_rmse_1d", val_edge_rmse_1d
                )

                combined_metric = val_edge_rmse_2d + val_edge_rmse_1d
                if self.early_stopping(combined_metric, self.model):
                    self.training_stats.log(
                        f"Early stopping triggered at epoch {epoch + 1}."
                    )
                    break

        self.training_stats.end_train()

    def validate(self):
        val_tester = EdgeAutoregressive1D2DTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device,
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        edge_rmse_2d = val_tester.get_avg_edge_rmse()
        edge_rmse_1d = val_tester.get_avg_edge_1d_rmse()
        return edge_rmse_2d, edge_rmse_1d

    def _compute_edge_loss(self, edge_pred: Tensor, label: Tensor) -> Tensor:
        return self.loss_func(edge_pred, label)

    def _override_pred_bc(self, edge_pred: Tensor, batch) -> Tensor:
        batch_boundary_edges_mask = np.tile(self.boundary_edges_mask, batch.num_graphs)
        edge_pred[batch_boundary_edges_mask] = batch.y_edge[batch_boundary_edges_mask].float()
        return edge_pred
