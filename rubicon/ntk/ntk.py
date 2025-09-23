"""Contains the implementations of NTK utilities."""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Iterator

import jax
from jax import grad, numpy as jnp
import neural_tangents as nt
import optax

from rubicon.nns.convnet import ConvNet
from rubicon.utils.jax import jax_cpu_backend
from rubicon.utils.kare import kare
from rubicon.utils.losses import cross_entropy_generic as cross_entropy


@dataclass
class NTKConfig:
    z: float = 1e-3
    optimizer: optax.GradientTransformationExtraArgs = optax.adam
    learning_rate: float = 1e-3
    lambd: float = 1e-6


@dataclass
class TrainingHistory:
    """Records the training history of the NTK via KARE."""
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    test_accuracy: list[float] = field(default_factory=list)


class NeuralTangentKernel:
    def __init__(
        self,
        params = None,
        kernel_fn = None,
        apply_fn = None,
        config: NTKConfig = NTKConfig()
    ) -> None:
        """Create an NTK instance.

        Args:
            params: The parameters of the model.
            kernel_fn: The kernel function of the model.
        """
        # the initial parameters. they are not supposed to be updated
        # contrary to the params
        self._init_params = params
        self.params = params
        # for now, i don't use this kernel function. but i keep it here
        # until i figure out how it can ease the current implementation.
        self.kernel_fn = kernel_fn
        self.apply_fn = apply_fn
        self.cfg = config
        # the kernel matrix is kept here because it's used in different
        # places in a single function, so it looks like a good idea to
        # not compute it multiple times, even though we could just get
        # it every time given the params
        self.K = None

    @property
    def init_params(self): return self._init_params
        
    @staticmethod
    def from_convnet(nn: ConvNet, config: NTKConfig = NTKConfig()) -> 'NeuralTangentKernel':
        """Create an NTK instance from a ConvNet instance.
        
        Args:
            nn: The ConvNet instance.
            config: Configurations of mostly the hyperparameters of the NTK.
        
        Returns:
            An NTK instance.
        """
        assert nn.initialized, \
            "ConvNet must have been initialized before extracting the NTK."
        return NeuralTangentKernel(
            params=nn.params,
            kernel_fn=nn.kernel_fn,
            apply_fn=nn.apply_fn,
            config=config
        )

    def train_with_kare(
        self,
        data_factory: Callable[[], tuple[Iterator[tuple[jnp.ndarray, jnp.ndarray]], Iterator[tuple[jnp.ndarray, jnp.ndarray]]]],
        num_epochs: int = 10,
        start_from_init: bool = False,
        return_metrics: bool = False,
        verbose: bool = False
    ) -> None:
        """Train the NTK instance using kernel ridge regression.
        
        Args:
            num_epochs: The number of epochs to train for.
            start_from_init: Whether to start training from the initial params
        """

        # This determines whether to consider starting with the initial set
        # of parameters or whatever happened to the point of this function
        # being called.
        params = self.init_params if start_from_init else self.params
        params = jax.tree_map(jnp.asarray, params)

        # @jax.jit
        def _kare_objective(x_train, y_train, params, z):
            K = self._compute_ntk(x_train, x_train, params, self.apply_fn)
            return kare(y_train, K, z)

        grad_kare = grad(_kare_objective)
        optimizer = self.cfg.optimizer(self.cfg.learning_rate)
        opt_state = optimizer.init(params)

        training_history = TrainingHistory()
        
        for epoch in range(num_epochs):
            train_iter, test_iter = data_factory()
            train_loss_sum, train_acc_sum, num_batches = 0.0, 0.0, 0

            for x_train, y_train in train_iter:
                # This is computed as the current kernel matrix. it's
                # redundant, but i don't yet have time to try to optimize it.
                self.K = self._compute_ntk(
                    x1=x_train,
                    x2=x_train,
                    params=params,
                    apply_fn=self.apply_fn
                )
                grads = grad_kare(x_train, y_train, params, self.cfg.z)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                if not (return_metrics or verbose):
                    continue
                batch_preds = self.predict(
                    K_train=self.K,
                    x_test=x_train,
                    x_train=x_train,
                    y_train=y_train,
                    params=params,
                    apply_fn=self.apply_fn
                )
                train_loss_sum += cross_entropy(batch_preds, y_train)
                train_acc_sum += accuracy(batch_preds, y_train)
                num_batches += 1
            
            train_loss_avg = train_loss_sum / num_batches or 0.0
            train_acc_avg = train_acc_sum / num_batches or 0.0

            if not (return_metrics or verbose):
                continue
            for x_test, y_test in test_iter:
                batch_preds = self.predict(
                    K_train=self.K,
                    x_test=x_test,
                    x_train=x_train,
                    y_train=y_train,
                    params=params,
                    apply_fn=self.apply_fn
                )
                test_loss_sum += cross_entropy(batch_preds, y_test)
                test_acc_sum += accuracy(batch_preds, y_test)
                num_batches += 1
            test_loss_avg = test_loss_sum / num_batches or 0.0
            test_acc_avg = test_acc_sum / num_batches or 0.0
       
            if return_metrics:
                training_history.epochs.append(epoch)
                training_history.train_loss.append(train_loss_avg)
                training_history.train_accuracy.append(train_acc_avg)
                training_history.test_loss.append(test_loss_avg)
                training_history.test_accuracy.append(test_acc_avg)

            if verbose:
                s = f"Epoch {epoch:<4} | Train loss: {train_loss_avg:.4f} |" \
                    f" Train accuracy: {train_acc_avg:.4f}"
                if test_iter is not None:
                    s += f" | Test loss: {test_loss_avg:.4f} |" \
                        f" Test accuracy: {test_acc_avg:.4f}"
                print(s)
        
        if return_metrics: return training_history

    @partial(jax.jit, static_argnums=(0, 4))
    @jax_cpu_backend
    def _compute_ntk(self, x1, x2, params, apply_fn):
        """Compute the NTK between two inputs."""
        ntk_fn = nt.empirical_ntk_fn(apply_fn)
        return ntk_fn(x1, x2, params)
    
    @partial(jax.jit, static_argnums=(0, 6))
    def predict(self, K_train, x_test, x_train, y_train, params, apply_fn):
        """Predicts the out of the Neural Tangent Kernel."""
        n = K_train.shape[0]
        K_mixed = self._compute_ntk(x_test, x_train, params, apply_fn)
        inv = jnp.linalg.inv((1 / n) * K_train + self.cfg.lambd * jnp.eye(n))
        return (1 / n) * K_mixed @ inv @ y_train
