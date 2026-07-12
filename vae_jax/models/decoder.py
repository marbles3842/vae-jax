import jax
import distrax
from flax import nnx


class DecoderNet(nnx.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_shape: tuple, rngs: nnx.Rngs, activation_fn=nnx.leaky_relu):
        self.output_shape = output_shape
        output_dim = 1
        for d in output_shape:
            output_dim *= d

        self.linear1 = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.activation_fn = activation_fn

    def __call__(self, z: jax.Array) -> jax.Array:
        x = self.activation_fn(self.linear1(z))
        x = self.activation_fn(self.linear2(x))
        x = self.linear3(x)
        return x.reshape(x.shape[:-1] + self.output_shape)


class BernoulliDecoder(nnx.Module):
    def __init__(self, decoder_net: nnx.Module):
        self.decoder_net = decoder_net

    def __call__(self, z: jax.Array) -> distrax.Distribution:
        logits = self.decoder_net(z)
        return distrax.Independent(
            distrax.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=len(self.decoder_net.output_shape)
        )
