from constants import EDGE_MODELS, NODE_EDGE_MODELS
from .base_trainer import BaseTrainer
from .dual_autoregressive_trainer import DualAutoregressiveTrainer
# from .dual_regression_trainer import DualRegressionTrainer
from .dual_regression_1d2d_trainer import DualRegression1D2DTrainer
# from .edge_autoregressive_trainer import EdgeAutoregressiveTrainer
from .edge_regression_1d2d_trainer import EdgeAutoregressive1D2DTester
# from .edge_regression_trainer import EdgeRegressionTrainer
from .edge_regression_1d2d_trainer import EdgeRegression1D2DTrainer
# from .node_autoregressive_trainer import NodeAutoregressiveTrainer
from .node_autoregressive_1d2d_trainer import NodeAutoregressive1D2DTrainer
# from .node_regression_trainer import NodeRegressionTrainer
from .node_regression_1d2d_trainer import NodeRegression1D2DTrainer

def trainer_factory(model_name: str, autoregressive: bool, *args, **kwargs) -> BaseTrainer:
    if model_name in NODE_EDGE_MODELS:
        if autoregressive:
            return DualAutoregressiveTrainer(*args, **kwargs)
        return DualRegression1D2DTrainer(*args, **kwargs)

    if model_name in EDGE_MODELS:
        if autoregressive:
            return EdgeAutoregressive1D2DTester(*args, **kwargs)
        return EdgeRegression1D2DTrainer(*args, **kwargs)

    if autoregressive:
        return NodeAutoregressive1D2DTrainer(*args, **kwargs)
    return NodeRegression1D2DTrainer(*args, **kwargs)

__all__ = [
    'DualAutoregressiveTrainer',
    'DualRegression1D2DTrainer',
    'EdgeRegression1D2DTrainer',
    'NodeAutoregressive1D2DTrainer',
    'NodeRegression1D2DTrainer',
    'trainer_factory',
]
