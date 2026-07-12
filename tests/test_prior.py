import jax.numpy as jnp
import jax.random as jr
import distrax
from vae_jax.models.prior import GaussianPrior, MixtureOfGaussians


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


def test_mixture_of_gaussians_initialization():
    latent_dim = 8
    key = jr.PRNGKey(0)
    num_components = 16
    prior = MixtureOfGaussians(latent_dim, key)

    assert prior.latent_dim == latent_dim
    assert prior.num_components == num_components
    assert prior.prior_means[...].shape == (num_components, latent_dim)
    assert prior.prior_stds[...].shape == (num_components, latent_dim)
    assert prior.weights[...].shape == (num_components,)
    assert jnp.all(prior.prior_stds[...] == 1.0)


def test_mixture_of_gaussians_custom_components():
    latent_dim = 8
    key = jr.PRNGKey(0)
    num_components = 5
    prior = MixtureOfGaussians(latent_dim, key, num_components=num_components)

    assert prior.num_components == num_components
    assert prior.prior_means[...].shape == (num_components, latent_dim)
    assert prior.prior_stds[...].shape == (num_components, latent_dim)
    assert prior.weights[...].shape == (num_components,)


def test_mixture_of_gaussians_distribution():
    latent_dim = 8
    key = jr.PRNGKey(0)
    prior = MixtureOfGaussians(latent_dim, key)
    dist = prior()

    assert isinstance(dist, distrax.Distribution)


def test_mixture_of_gaussians_sampling():
    latent_dim = 8
    key = jr.PRNGKey(0)
    prior = MixtureOfGaussians(latent_dim, key)
    dist = prior()

    sample_key = jr.PRNGKey(1)
    sample = dist.sample(seed=sample_key)
    assert sample.shape == (latent_dim,)

    batch_samples = dist.sample(seed=sample_key, sample_shape=(10,))
    assert batch_samples.shape == (10, latent_dim)


def test_mixture_of_gaussians_log_prob():
    latent_dim = 8
    key = jr.PRNGKey(0)
    prior = MixtureOfGaussians(latent_dim, key)
    dist = prior()

    sample_key = jr.PRNGKey(1)
    sample = dist.sample(seed=sample_key)
    log_prob = dist.log_prob(sample)
    assert log_prob.shape == ()

    batch_samples = dist.sample(seed=sample_key, sample_shape=(10,))
    batch_log_prob = dist.log_prob(batch_samples)
    assert batch_log_prob.shape == (10,)
