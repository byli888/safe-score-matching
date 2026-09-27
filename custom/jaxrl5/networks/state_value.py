import flax.linen as nn
import jax.numpy as jnp

from jaxrl5.networks import default_init


class StateValue(nn.Module):
    base_cls: nn.Module
    out_dim: int = 1

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(self.out_dim, kernel_init=default_init(), name="OutputVDense")(outputs)
        if self.out_dim == 1:
            return jnp.squeeze(value, -1)
        return value
    
class Relu_StateValue(nn.Module):
    base_cls: nn.Module
    out_dim: int = 1

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        value = nn.Dense(self.out_dim, kernel_init=default_init(), name="OutputVDense")(outputs)

        value = nn.softplus(value)
        if self.out_dim == 1:
            return jnp.squeeze(value, -1)
        return value
