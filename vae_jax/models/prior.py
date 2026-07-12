import jax.numpy as jnp
import distrax
from flax import nnx


class GaussianPrior(nnx.Module):

    def __init__(self, M: int):
        self.M = M
        self.mean = nnx.Variable(jnp.zeros(M))
        self.std  = nnx.Variable(jnp.ones(M))

    def __call__(self) -> distrax.Distribution:
        return distrax.Independent(
            distrax.Normal(loc=self.mean[...], scale=self.std[...]),
            reinterpreted_batch_ndims=1,
        )


# TODO: provide proper implementation

class MixtureOfGaussians(nnx.Module):

    def __init__(self, K: int, latent_dim: int):
        self.latent_dim = latent_dim

    def __call__(self) -> distrax.Distribution:
        return GaussianPrior(self.latent_dim)