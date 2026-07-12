import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from vae_jax.models.prior import GaussianPrior
from vae_jax.models.encoder import EncoderNet, GaussianEncoder
from vae_jax.models.decoder import DecoderNet, BernoulliDecoder
from vae_jax.models.vae import VAE


def test_vae():
    # pylint: disable=too-many-locals
    batch_size = 4
    latent_dim = 8
    hidden_dim = 16
    img_shape = (28, 28)

    rngs = nnx.Rngs(0)

    # Initialize components
    prior = GaussianPrior(latent_dim)
    encoder_net = EncoderNet(img_shape[0] * img_shape[1], hidden_dim, latent_dim, rngs=rngs)
    decoder_net = DecoderNet(latent_dim, hidden_dim, img_shape, rngs=rngs)
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(decoder_net)

    # Create VAE
    vae = VAE(prior, encoder, decoder)

    # Prepare inputs and rng
    x = jnp.zeros((batch_size, *img_shape))
    key = jr.PRNGKey(0)
    key, rng_elbo = jr.split(key)

    # Test elbo
    elbo_val = vae.elbo(x, rng_elbo)
    assert isinstance(elbo_val, jax.Array)
    assert elbo_val.shape == ()

    # Test __call__
    loss_val = vae(x, rng_elbo)
    assert isinstance(loss_val, jax.Array)
    assert loss_val.shape == ()
    assert jnp.allclose(loss_val, -elbo_val)

    # Test sample
    n_samples = 10
    samples = vae.sample(key, n_samples=n_samples)
    assert samples.shape == (n_samples, *img_shape)
