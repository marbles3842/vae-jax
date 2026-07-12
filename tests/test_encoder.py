import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import distrax
from vae_jax.models.encoder import EncoderNet, GaussianEncoder


def test_encoder_net():
    batch_size = 4
    input_dim = 28 * 28
    hidden_dim = 16
    latent_dim = 8

    # Initialize RNGs
    rngs = nnx.Rngs(0)

    # Initialize encoder net
    encoder_net = EncoderNet(input_dim, hidden_dim, latent_dim, rngs=rngs)

    # Test with flat input
    x_flat = jnp.zeros((batch_size, input_dim))
    out_flat = encoder_net(x_flat)
    assert out_flat.shape == (batch_size, latent_dim * 2)

    # Test with multi-dimensional input (should reshape to flat automatically)
    x_grid = jnp.zeros((batch_size, 28, 28))
    out_grid = encoder_net(x_grid)
    assert out_grid.shape == (batch_size, latent_dim * 2)


def test_gaussian_encoder():
    batch_size = 4
    input_dim = 28 * 28
    hidden_dim = 16
    latent_dim = 8

    rngs = nnx.Rngs(0)
    encoder_net = EncoderNet(input_dim, hidden_dim, latent_dim, rngs=rngs)
    encoder = GaussianEncoder(encoder_net)

    x = jnp.zeros((batch_size, 28, 28))
    dist = encoder(x)

    assert isinstance(dist, distrax.Distribution)

    # Test sample and log_prob shapes
    key = jr.PRNGKey(0)
    samples = dist.sample(seed=key)
    assert samples.shape == (batch_size, latent_dim)

    log_prob = dist.log_prob(samples)
    assert log_prob.shape == (batch_size,)
