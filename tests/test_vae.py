import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from vae_jax.models.prior import GaussianPrior
from vae_jax.models.encoder import EncoderNet, GaussianEncoder
from vae_jax.models.decoder import DecoderNet, BernoulliDecoder
from vae_jax.models.vae import VAE


@pytest.fixture(name="vae_config")
def fixture_vae_config():
    return {
        "batch_size": 4,
        "latent_dim": 8,
        "hidden_dim": 16,
        "img_shape": (28, 28)
    }


@pytest.fixture(name="vae_model")
def fixture_vae_model(vae_config):
    rngs = nnx.Rngs(0)
    img_shape = vae_config["img_shape"]
    latent_dim = vae_config["latent_dim"]
    hidden_dim = vae_config["hidden_dim"]

    prior = GaussianPrior(latent_dim)
    encoder_net = EncoderNet(img_shape[0] * img_shape[1], hidden_dim, latent_dim, rngs=rngs)
    decoder_net = DecoderNet(latent_dim, hidden_dim, img_shape, rngs=rngs)
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(decoder_net)

    return VAE(prior, encoder, decoder)


def test_vae_elbo(vae_model, vae_config):
    batch_size = vae_config["batch_size"]
    img_shape = vae_config["img_shape"]

    x = jnp.zeros((batch_size, *img_shape))
    rng_elbo = jr.PRNGKey(1)

    elbo_val = vae_model.elbo(x, rng_elbo)
    assert isinstance(elbo_val, jax.Array)
    assert elbo_val.shape == ()

    assert jnp.allclose(elbo_val, -565.2810668945312, atol=1e-3)


def test_vae_loss(vae_model, vae_config):
    batch_size = vae_config["batch_size"]
    img_shape = vae_config["img_shape"]

    x = jnp.zeros((batch_size, *img_shape))
    rng_elbo = jr.PRNGKey(1)

    loss_val = vae_model(x, rng_elbo)

    assert isinstance(loss_val, jax.Array)
    assert loss_val.shape == ()

    assert jnp.allclose(loss_val, 565.2810668945312, atol=1e-3)


def test_vae_sample(vae_model, vae_config):
    img_shape = vae_config["img_shape"]
    n_samples = 10
    key = jr.PRNGKey(2)

    samples = vae_model.sample(key, n_samples=n_samples)
    assert samples.shape == (n_samples, *img_shape)

    assert jnp.all((samples == 0.0) | (samples == 1.0))
