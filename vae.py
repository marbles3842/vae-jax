# Flax NNX port of the Bernoulli VAE
# Original PyTorch version: DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Inspiration:
#   - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
#   - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import jax
import jax.numpy as jnp
from jax import random
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



class EncoderNet(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, latent_dim * 2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        return self.linear3(x)


class GaussianEncoder(nnx.Module):
    """
    Gaussian encoder  q(z | x) = N(mu(x), sigma^2(x)).

    Parameters
    ----------
    encoder_net : nnx.Module
        Network that maps x -> (batch, 2M); first M outputs are the mean,
        last M are the log-standard-deviations.
    """

    def __init__(self, encoder_net: nnx.Module):
        self.encoder_net = encoder_net

    def __call__(self, x: jax.Array) -> distrax.Distribution:
        """
        Parameters
        ----------
        x : jax.Array  shape (batch, H, W)  or  (batch, D)

        Returns
        -------
        distrax.Independent[distrax.Normal] -- approximate posterior q(z|x)
        """
        out = self.encoder_net(x)                      # (batch, 2M)
        mean, log_std = jnp.split(out, 2, axis=-1)
        return distrax.Independent(
            distrax.Normal(loc=mean, scale=jnp.exp(log_std)),
            reinterpreted_batch_ndims=1,
        )


# ---------------------------------------------------------------------------
# Decoder network
# ---------------------------------------------------------------------------

class DecoderNet(nnx.Module):
    """MLP: latent code z -> flat logits, reshaped to (H, W)."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_shape: tuple, rngs: nnx.Rngs):
        self.output_shape = output_shape          # e.g. (28, 28)
        output_dim = 1
        for d in output_shape:
            output_dim *= d

        self.linear1 = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        x = nnx.relu(self.linear1(z))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)                       # (batch, H*W)
        # x = x.reshape(x.shape[0], *self.output_shape)
        return x


class BernoulliDecoder(nnx.Module):
    """
    Bernoulli decoder  p(x | z) = Bernoulli(sigma(logits(z))).

    Parameters
    ----------
    decoder_net : nnx.Module
        Network that maps z -> logits with the same shape as the data.
    """

    def __init__(self, decoder_net: nnx.Module):
        self.decoder_net = decoder_net

    def __call__(self, z: jax.Array) -> distrax.Distribution:
        """
        Parameters
        ----------
        z : jax.Array  shape (batch, M)

        Returns
        -------
        distrax.Independent[distrax.Bernoulli] -- likelihood p(x|z)
        """
        logits = self.decoder_net(z)              # (batch, H, W)
        return distrax.Independent(
            distrax.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=1,          # sum over flattened D=784
        )


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

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
