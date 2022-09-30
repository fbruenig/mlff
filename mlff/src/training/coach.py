import jax.numpy as jnp
import logging

from dataclasses import dataclass
from typing import (Any, Callable, Dict, Sequence, Tuple)
from flax.core.frozen_dict import FrozenDict
from functools import partial

from mlff.src.training.stopping_criteria import stop_by_lr, stop_by_metric
from .run import run_training

logging.basicConfig(level=logging.INFO)

Array = Any
StackNet = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]


@dataclass
class Coach:
    input_keys: Sequence[str]
    target_keys: Sequence[str]
    epochs: int
    training_batch_size: int
    validation_batch_size: int
    loss_weights: Dict[str, float]
    ckpt_dir: str
    data_path: str
    net_seed: int = 0
    training_seed: int = 0
    stop_lr_min: float = None
    stop_metrics_min: Dict[str, float] = None

    def run(self, train_state, train_ds, valid_ds, loss_fn, metric_fn=None, **kwargs):
        if self.stop_lr_min is not None:
            stop_lr_fn = partial(stop_by_lr, lr_min=self.stop_lr_min)
        else:
            stop_lr_fn = None

        if self.stop_metrics_min is not None:
            stop_metric_fn = partial(stop_by_metric, target_metrics=self.stop_metrics_min)
        else:
            stop_metric_fn = None

        run_training(state=train_state,
                     loss_fn=loss_fn,
                     metric_fn=metric_fn,
                     train_ds=train_ds,
                     valid_ds=valid_ds,
                     epochs=self.epochs,
                     train_bs=self.training_batch_size,
                     valid_bs=self.validation_batch_size,
                     ckpt_dir=self.ckpt_dir,
                     stop_lr_fn=stop_lr_fn,
                     stop_metric_fn=stop_metric_fn,
                     seed=self.training_seed,
                     **kwargs
                     )

    def __dict_repr__(self):
        return {'coach': {'input_keys': self.input_keys,
                          'target_keys': self.target_keys,
                          'epochs': self.epochs,
                          'training_batch_size': self.training_batch_size,
                          'validation_batch_size': self.validation_batch_size,
                          'loss_weights': self.loss_weights,
                          'ckpt_dir': self.ckpt_dir,
                          'data_path': self.data_path,
                          'training_seed': self.training_seed,
                          'net_seed': self.net_seed,
                          'stop_lr_min': self.stop_lr_min,
                          'stop_metrics_min': self.stop_metrics_min}}
