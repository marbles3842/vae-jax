import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
from vae_jax.models.prior import MixtureOfGaussians
from vae_jax.models.encoder import GaussianEncoder
from vae_jax.models.decoder import BernoulliDecoder


class VAE(nnx.Module):
    def __init__(self, prior: nnx.Module, encoder: GaussianEncoder, decoder: BernoulliDecoder):
        self.prior   = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, x: jax.Array, rng: jax.Array) -> jax.Array:
        q = self.encoder(x)
        z = q.sample(seed=rng)

        RE = self.decoder(z).log_prob(x)
        KL = q.log_prob(z) - self.prior().log_prob(z)
        return (RE - KL).mean()

        # RE = self.decoder(z).log_prob(x)
        # KL_divergence = q.log_prob(z) - self.prior().log_prob(z) 
        # return (RE - KL_divergence).mean()
        

        # q = self.encoder(x)
        # z = q.sample(seed=rng)
        # log_px_z = self.decoder(z).log_prob(x)
        
        # prior_log_prob = self.prior().log_prob(z)
        # encoder_log_prob = q.log_prob(z)
        # kl = prior_log_prob - encoder_log_prob
        # jax.debug.print("prior_log_prob mean: {}, encoder_log_prob mean: {}, KL mean: {}", prior_log_prob.mean(), encoder_log_prob.mean(), kl.mean())
        # return jnp.mean(log_px_z - kl)

    def sample(self, rng: jax.Array, n_samples: int = 1) -> jax.Array:
        rng_z, rng_x = random.split(rng)
        z = self.prior().sample(seed=rng_z, sample_shape=(n_samples,))
        return self.decoder(z).sample(seed=rng_x)

    def __call__(self, x: jax.Array, rng: jax.Array) -> jax.Array:
        return -self.elbo(x, rng)
