import jax.numpy as jnp
import distrax
from flax import nnx


class MixtureOfGaussians(nnx.Module):

    def __init__(self, M: int, num_components: int):
        self.M = M
        self.num_components = num_components
        
        self.means = nnx.Variable(jnp.zeros((num_components, M)))
        self.stds  = nnx.Variable(jnp.ones((num_components, M)))
        
    def __call__(self):
        return distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(logits=jnp.ones(self.num_components)),
            components_distribution=distrax.Independent(
                distrax.Normal(loc=self.means[...], scale=self.stds[...]),
                reinterpreted_batch_ndims=1,
            ),
        )


class GaussianPrior(nnx.Module):

    def __init__(self, M: int):
        """
        Parameters
        ----------
        M : int
            Dimension of the latent space.
        """
        self.M = M
        self.mean = nnx.Variable(jnp.zeros(M))
        self.std  = nnx.Variable(jnp.ones(M))

    def __call__(self) -> distrax.Distribution:
        return distrax.Independent(
            distrax.Normal(loc=self.mean[...], scale=self.std[...]),
            reinterpreted_batch_ndims=1,
        )
