import jax
from jax import random
from flax import nnx
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

    def sample(self, rng: jax.Array, n_samples: int = 1) -> jax.Array:
        rng_z, rng_x = random.split(rng)
        z = self.prior().sample(seed=rng_z, sample_shape=(n_samples,))
        return self.decoder(z).sample(seed=rng_x)

    def __call__(self, x: jax.Array, rng: jax.Array) -> jax.Array:
        return -self.elbo(x, rng)
