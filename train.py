import jax
import jax.random as jr
from flax import nnx
from flax.nnx import jit
from tqdm import tqdm

from vae_jax import (
    VAE,
    GaussianPrior,
    MixtureOfGaussians,
    VampPrior,
    EncoderNet,
    GaussianEncoder,
    BernoulliDecoder,
    DecoderNet,
    MNISTInfo,
)


def init_model(rngs: nnx.Rngs, img_shape: tuple[int, int], latent_dim: int, hidden_dim: int, prior_type: str = "gauss"):
    encoder_net = EncoderNet(img_shape[0] * img_shape[1], hidden_dim, latent_dim, rngs=rngs)
    encoder     = GaussianEncoder(encoder_net)

    if prior_type == "mog":
        prior = MixtureOfGaussians(latent_dim=latent_dim, key=rngs.params())
    elif prior_type == "vamp":
        prior = VampPrior(
            latent_dim=latent_dim,
            input_shape=img_shape[0] * img_shape[1],
            key=rngs.params(),
            encoder=encoder,
        )
    else:
        prior = GaussianPrior(latent_dim)

    decoder_net = DecoderNet(latent_dim, hidden_dim, img_shape, rngs=rngs)
    decoder     = BernoulliDecoder(decoder_net)
    model       = VAE(prior, encoder, decoder)
    return model


@jit
def train_step(model: VAE, optimizer: nnx.Optimizer, rng_elbo: jax.Array, batch: jax.Array):
    def loss_fn(model, rng_elbo, x):
        return model(x, rng_elbo)

    loss, grads = nnx.value_and_grad(loss_fn)(model, rng_elbo, batch)
    optimizer.update(model, grads)
    return loss


def train(model: VAE, optimizer: nnx.Optimizer, mnist_dataset, epochs: int, batch_size: int, key: jax.Array) -> jax.Array:
    num_steps_per_epoch = MNISTInfo.train_length // batch_size
    total_steps = num_steps_per_epoch * epochs

    progress_bar = tqdm(enumerate(mnist_dataset), total=total_steps, desc="Training")

    for step, batch in progress_bar:
        batch = jax.device_put(batch)
        key, rng_elbo = jr.split(key)
        loss = train_step(model, optimizer, rng_elbo, batch)

        epoch = step // num_steps_per_epoch + 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch}/{epochs}")

    return key
