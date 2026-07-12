import jax.numpy as jnp
import jax.random as jr
import distrax
from vae_jax.models.prior import GaussianPrior


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
