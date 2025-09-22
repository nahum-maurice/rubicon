"""Contains the implementations of NTK utilities."""

from dataclasses import dataclass

import jax
from jax import grad, numpy as jnp
import neural_tangents as nt
import optax

from rubicon.nns import ConvNet
from rubicon.utils.kare import kare
from rubicon.utils.losses import cross_entropy


@dataclass
class NTKConfig:
    z: float = 1e-3
    optimizer: optax.GradientTransformationExtraArgs = optax.adam
    learning_rate: float = 1e-3
    lambd: float = 1e-6


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
        self.kernel_fn = kernel_fn
        self.apply_fn = apply_fn
        self.cfg = config

    @property
    def init_params(self): return self._init_params
        
    @staticmethod
    def from_convnet(nn: ConvNet) -> 'NeuralTangentKernel':
        """Create an NTK instance from a ConvNet instance.
        
        Args:
            nn: The ConvNet instance.
        
        Returns:
            An NTK instance.
        """
        return NeuralTangentKernel(
            params=nn.params,
            kernel_fn=nn.kernel_fn,
            apply_fn=nn.apply_fn,
            config=nn.cfg
        )

    def train_with_kare(
        self,
        num_epochs: int = 10,
        steps_per_epoch: int = 100,
        start_from_init: bool = False,
        return_metrics: bool = False,
        verbose: bool = False
    ) -> None:
        """Train the NTK instance using kernel ridge regression.
        
        Args:
            num_epochs: The number of epochs to train for.
            start_from_init: Whether to start training from the initial params
        """
        @jax.jit
        def _kare_objective(x_train, y_train):
            K = self._compute_ntk(x_train, x_train, self.params)
            return kare(y_train, K, self.cfg.z)
        
        grad_kare = grad(_kare_objective)
        optimizer = self.cfg.optimizer(self.cfg.learning_rate)
        if start_from_init:
            opt_state = optimizer.init(self.init_params)
        else:
            opt_state = optimizer.init(self.params)
        
        for epoch in range(num_epochs):
            train_loss_sum, train_acc_sum, num_batches = 0

            for step in range(steps_per_epoch):
                grads = grad_kare(self.params)
                updates, opt_state = optimizer.update(grads, opt_state)
                self.params = optax.apply_updates(self.params, updates)   

    @jax.jit
    def _compute_ntk(self, x1, x2):
        """Compute the NTK between two inputs."""
        ntk_fn = nt.empirical_ntk_fn(self.apply_fn)
        return ntk_fn(x1, x2, self.params)
    
    @jax.jit
    def predict(self, K_train, x_test, x_train, y_train):
        """Predicts the out of the Neural Tangent Kernel. 
        
        NTK-KARE(x) = n^{-1} K(x,X)(n^{-1} K(X,X) + \lambda I_n)^{-1} y
        """
        n = K_train.shape[0]
        K_mixed = self._compute_ntk(x_test, x_train)
        inv = jnp.linalg.inv((1 / n) * K_train + self.cfg.lambd * jnp.eye(n))
        return (1 / n) * K_mixed @ inv @ y_train
