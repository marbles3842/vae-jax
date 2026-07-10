import jax
import distrax
from flax import nnx


class DecoderNet(nnx.Module):
    """MLP: latent code z -> flat logits, reshaped to (H, W)."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_shape: tuple, rngs: nnx.Rngs):
        self.output_shape = output_shape          # e.g. (28, 28)
        output_dim = 1
        for d in output_shape:
            output_dim *= d

        self.linear1 = nnx.Linear(latent_dim, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        x = nnx.relu(self.linear1(z))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)                       # (batch, H*W)
        return x.reshape(x.shape[:-1] + self.output_shape)


class BernoulliDecoder(nnx.Module):
    """
    Bernoulli decoder  p(x | z) = Bernoulli(sigma(logits(z))).

    Parameters
    ----------
    decoder_net : nnx.Module
        Network that maps z -> logits with the same shape as the data.
    """

    def __init__(self, decoder_net: nnx.Module):
        self.decoder_net = decoder_net

    def __call__(self, z: jax.Array) -> distrax.Distribution:
        """
        Parameters
        ----------
        z : jax.Array  shape (batch, M)

        Returns
        -------
        distrax.Independent[distrax.Bernoulli] -- likelihood p(x|z)
        """
        logits = self.decoder_net(z)              # (batch, H, W)
        return distrax.Independent(
            distrax.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=len(self.decoder_net.output_shape),          # sum over spatial dimensions
        )
