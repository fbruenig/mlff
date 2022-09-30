import jax.numpy as jnp
from flax import linen as nn
from typing import (Callable, Sequence)


class MLP(nn.Module):
    """ Standard implementation of an MLP.
    """
    features: Sequence[int]
    """ Dimensions of the features in the layers. Length of the passed sequence equals the
        number of layers in the net. """

    activation_fn: Callable = lambda x: x
    """ activation function to use in the network. """

    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, use_bias=self.use_bias, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = self.activation_fn(x)
        return x
