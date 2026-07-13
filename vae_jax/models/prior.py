import jax
import distrax
import jax.numpy as jnp
import distrax
import jax.random as jr
from flax import nnx

_MOG_DEFAULT_NUM_COMPONENTS = 16

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


class MixtureOfGaussians(nnx.Module):

    def __init__(self, latent_dim: int, key: jax.Array, num_components: int = _MOG_DEFAULT_NUM_COMPONENTS):
        self.latent_dim = latent_dim
        self.num_components = num_components

        key_means, key_weights = jr.split(key, 2)

        self.prior_means  = nnx.Param(jr.normal(key_means, shape=(num_components, latent_dim)))
        self.raw_prior_stds = nnx.Param(jnp.zeros((num_components, latent_dim)))
        self.weights = nnx.Param(jr.normal(key_weights, shape=(num_components, )))


    def __call__(self) -> distrax.Distribution:

        mixture_d = distrax.Categorical(probs = nnx.softmax(self.weights.value, axis=0))
        stds = jax.nn.softplus(self.raw_prior_stds.value) + 1e-6
        component_d = distrax.Independent(
            distrax.Normal(loc=self.prior_means.value, scale=stds),
            reinterpreted_batch_ndims=1,
        )

        return distrax.MixtureSameFamily(
            mixture_distribution = mixture_d,
            components_distribution = component_d
        )

class VampPrior(nnx.Module):
    def __init__(self, latent_dim: int, input_shape: int, key: jax.Array, encoder: nnx.Module, num_components: int = _MOG_DEFAULT_NUM_COMPONENTS):
        self.latent_dim = latent_dim
        self.num_components = num_components

        self.encoder = encoder

        key_weights, key_inputs = jr.split(key, 2)
        self.pseudoinputs = nnx.Param(jr.normal(key_inputs, shape=(num_components, input_shape)))
        self.weights = nnx.Param(jr.normal(key_weights, shape=(num_components, )))



    def __call__(self) -> distrax.Distribution:

        component_d = self.encoder(self.pseudoinputs.value)
        mixture_d = distrax.Categorical(probs = nnx.softmax(self.weights.value, axis=0))
        
        return distrax.MixtureSameFamily(
            mixture_distribution = mixture_d,
            components_distribution = component_d
        )