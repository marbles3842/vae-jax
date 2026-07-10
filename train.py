from tensorflow_probability.substrates import jax
import jax
import optax
import jax.random as jr
import matplotlib.pyplot as plt

from flax import nnx
from flax.nnx import jit

from vae_jax import (
    VAE,
    GaussianPrior,
    MixtureOfGaussians,
    EncoderNet,
    GaussianEncoder,
    BernoulliDecoder,
    DecoderNet,
    get_mnist_dataset,
    MNISTInfo,
)


def init_model(rngs: nnx.Rngs, img_shape: tuple[int, int], latent_dim: int, hidden_dim: int):
    prior       = GaussianPrior(latent_dim)
    # prior = MixtureOfGaussians(latent_dim, 6)
    encoder_net = EncoderNet(img_shape[0] * img_shape[1], hidden_dim, latent_dim, rngs=rngs)
    decoder_net = DecoderNet(latent_dim, hidden_dim, img_shape, rngs=rngs)
    encoder     = GaussianEncoder(encoder_net)
    decoder     = BernoulliDecoder(decoder_net)
    model       = VAE(prior, encoder, decoder)
    return model


def save_samples(samples: jax.Array, n_samples: int, filename: str = "samples.png"):
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


@jit
def train_step(model: VAE, optimizer: nnx.Optimizer, rng_elbo: jax.Array, batch: jax.Array):
    def loss_fn(model, rng_elbo, x):
        return model(x, rng_elbo)

    loss, grads = nnx.value_and_grad(loss_fn)(model, rng_elbo, batch)
    optimizer.update(model, grads)
    return loss



if __name__ == "__main__":
    # -- hyper-parameters --------------------------------------------------
    BATCH_SIZE     = 128
    IMG_H, IMG_W, N_CHANNELS   = 28, 28, 1
    LATENT_DIM     = 10
    HIDDEN_DIM     = 512
    EPOCHS         = 5
    N_SAMPLES      = 10

    seed = 0
    key  = jr.PRNGKey(seed)

    # -- build model -------------------------------------------------------
    rngs = nnx.Rngs(params=key)

    model = init_model(rngs, (IMG_H, IMG_W), LATENT_DIM, HIDDEN_DIM)

    print("Model built successfully.\n")

    key, sample_key = jr.split(key)

    mnist_dataset = get_mnist_dataset(train_batch_size=BATCH_SIZE, num_epochs=EPOCHS, seed=seed, train_worker_count=4)

    print('Training started')

    epoch = 0

    num_steps_per_epoch = MNISTInfo.train_length // BATCH_SIZE

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    for step, batch in enumerate(mnist_dataset):
        batch = jax.device_put(batch)
        key, rng_elbo = jr.split(key)
        loss = train_step(model, optimizer, rng_elbo, batch)

        if (step + 1) % num_steps_per_epoch == 0:
            epoch += 1
            print(f"epoch: {epoch}, loss: {loss.item():.4f}")

    samples = model.sample(n_samples=N_SAMPLES, rng=sample_key)

    save_samples(samples, N_SAMPLES, "samples.png")

    print('Training completed')