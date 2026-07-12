import jax
import jax.numpy as jnp
import distrax
from flax import nnx


class EncoderNet(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, rngs: nnx.Rngs, activation_fn=nnx.leaky_relu):
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, latent_dim * 2, rngs=rngs)
        self.activation_fn = activation_fn

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(x.shape[0], -1)
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        return self.linear3(x)


class GaussianEncoder(nnx.Module):

    def __init__(self, encoder_net: nnx.Module):
        self.encoder_net = encoder_net

    def __call__(self, x: jax.Array) -> distrax.Distribution:

        out = self.encoder_net(x)
        mean, log_std = jnp.split(out, 2, axis=-1)
        return distrax.Independent(
            distrax.Normal(loc=mean, scale=jnp.exp(log_std)),
            reinterpreted_batch_ndims=1,
        )
