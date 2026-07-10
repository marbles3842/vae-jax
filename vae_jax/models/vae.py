import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
from vae_jax.models.prior import GaussianPrior
from vae_jax.models.encoder import GaussianEncoder
from vae_jax.models.decoder import BernoulliDecoder


class VAE(nnx.Module):
    """Variational Autoencoder with a Gaussian prior and Bernoulli likelihood."""

    def __init__(self, prior: GaussianPrior, encoder: GaussianEncoder, decoder: BernoulliDecoder):
        """
        Parameters
        ----------
        prior   : GaussianPrior
        encoder : GaussianEncoder
        decoder : BernoulliDecoder
        """
        self.prior   = prior
        self.encoder = encoder
        self.decoder = decoder

    def elbo(self, x: jax.Array, rng: jax.Array) -> jax.Array:
        """
        Monte-Carlo estimate of the ELBO averaged over the batch.

            ELBO = E_q[log p(x|z)] - KL( q(z|x) || p(z) )

        Parameters
        ----------
        x   : jax.Array  shape (batch, H, W)
        rng : jax.Array  PRNGKey used for the reparameterisation trick

        Returns
        -------
        Scalar ELBO value.
        """
        q = self.encoder(x)                           # q(z|x)
        z = q.sample(seed=rng)                        # reparameterised (batch, M)
        log_px_z = self.decoder(z).log_prob(x)        # (batch,)
        kl = q.kl_divergence(self.prior())            # (batch,)
        return jnp.mean(log_px_z - kl)

    def sample(self, rng: jax.Array, n_samples: int = 1) -> jax.Array:
        """
        Draw samples by ancestral sampling: z ~ p(z), x ~ p(x|z).

        Parameters
        ----------
        rng       : jax.Array  PRNGKey
        n_samples : int

        Returns
        -------
        jax.Array  shape (n_samples, H, W)  -- binary samples
        """
        rng_z, rng_x = random.split(rng)
        z = self.prior().sample(seed=rng_z, sample_shape=(n_samples,))  # (n, M)
        return self.decoder(z).sample(seed=rng_x)                        # (n, H, W)

    def __call__(self, x: jax.Array, rng: jax.Array) -> jax.Array:
        """Return the negative ELBO (the training loss)."""
        return -self.elbo(x, rng)
