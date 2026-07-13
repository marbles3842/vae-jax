import jax.numpy as jnp
import jax.random as jr
from flax import nnx
import distrax
from vae_jax.models.prior import GaussianPrior, MixtureOfGaussians, VampPrior
from vae_jax.models.encoder import EncoderNet, GaussianEncoder


def test_gaussian_prior():
    latent_dim = 8
    prior = GaussianPrior(latent_dim)

    assert prior.mean[...].shape == (latent_dim,)
    assert prior.std[...].shape == (latent_dim,)
    assert jnp.all(prior.mean[...] == 0.0)
    assert jnp.all(prior.std[...] == 1.0)

    dist = prior()
    assert isinstance(dist, distrax.Distribution)

    key = jr.PRNGKey(0)
    sample = dist.sample(seed=key)
    assert sample.shape == (latent_dim,)

    log_prob = dist.log_prob(sample)
    assert log_prob.shape == ()


def test_mixture_of_gaussians_init():
    latent_dim = 8
    num_components = 4
    key = jr.PRNGKey(0)

    prior = MixtureOfGaussians(latent_dim, key, num_components=num_components)

    assert prior.prior_means[...].shape == (num_components, latent_dim)
    assert prior.raw_prior_stds[...].shape == (num_components, latent_dim)
    assert prior.weights[...].shape == (num_components,)


def test_mixture_of_gaussians_distribution():
    latent_dim = 8
    num_components = 4
    key = jr.PRNGKey(0)

    prior = MixtureOfGaussians(latent_dim, key, num_components=num_components)
    dist = prior()

    assert isinstance(dist, distrax.Distribution)

    sample = dist.sample(seed=jr.PRNGKey(1))
    assert sample.shape == (latent_dim,)

    log_prob = dist.log_prob(sample)
    assert log_prob.shape == ()


def test_vamp_prior_init():
    latent_dim = 8
    input_shape = 16
    num_components = 4
    key = jr.PRNGKey(0)

    rngs = nnx.Rngs(1)
    encoder_net = EncoderNet(input_shape, 12, latent_dim, rngs=rngs)
    encoder = GaussianEncoder(encoder_net)

    prior = VampPrior(latent_dim, input_shape, key, encoder, num_components=num_components)

    assert prior.pseudoinputs[...].shape == (num_components, input_shape)
    assert prior.weights[...].shape == (num_components,)


def test_vamp_prior_distribution():
    latent_dim = 8
    input_shape = 16
    num_components = 4
    key = jr.PRNGKey(0)

    rngs = nnx.Rngs(1)
    encoder_net = EncoderNet(input_shape, 12, latent_dim, rngs=rngs)
    encoder = GaussianEncoder(encoder_net)

    prior = VampPrior(latent_dim, input_shape, key, encoder, num_components=num_components)
    dist = prior()

    assert isinstance(dist, distrax.Distribution)

    sample = dist.sample(seed=jr.PRNGKey(2))
    assert sample.shape == (latent_dim,)

    log_prob = dist.log_prob(sample)
    assert log_prob.shape == ()
