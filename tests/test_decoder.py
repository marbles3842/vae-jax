import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import distrax
from vae_jax.models.decoder import DecoderNet, BernoulliDecoder


def test_decoder_net():
    batch_size = 4
    latent_dim = 8
    hidden_dim = 16
    output_shape = (28, 28)

    rngs = nnx.Rngs(0)
    decoder_net = DecoderNet(latent_dim, hidden_dim, output_shape, rngs=rngs)

    z = jnp.zeros((batch_size, latent_dim))
    out = decoder_net(z)

    assert out.shape == (batch_size, 28, 28)


def test_bernoulli_decoder():
    batch_size = 4
    latent_dim = 8
    hidden_dim = 16
    output_shape = (28, 28)

    rngs = nnx.Rngs(0)
    decoder_net = DecoderNet(latent_dim, hidden_dim, output_shape, rngs=rngs)
    decoder = BernoulliDecoder(decoder_net)

    z = jnp.zeros((batch_size, latent_dim))
    dist = decoder(z)

    assert isinstance(dist, distrax.Distribution)

    # Test sample and log_prob shapes
    key = jr.PRNGKey(0)
    samples = dist.sample(seed=key)
    assert samples.shape == (batch_size, 28, 28)

    # Sample targets for log_prob should be of the same shape
    log_prob = dist.log_prob(samples)
    assert log_prob.shape == (batch_size,)
