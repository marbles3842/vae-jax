import os
import argparse
import jax
import jax.random as jr
import optax
from flax import nnx
from flax import serialization

from vae_jax import get_mnist_dataset
from train import init_model, train
from sample import sample


def save_checkpoint(model: nnx.Module, filepath: str):
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    state = nnx.state(model)
    state_dict = state.to_pure_dict()
    state_bytes = serialization.to_bytes(state_dict)
    with open(filepath, "wb") as f:
        f.write(state_bytes)
    print(f"Model state saved to {filepath}")


def load_checkpoint(model: nnx.Module, filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file {filepath} not found.")

    state = nnx.state(model)
    with open(filepath, "rb") as f:
        state_bytes = f.read()
    pure_dict = state.to_pure_dict()
    restored_dict = serialization.from_bytes(pure_dict, state_bytes)
    restored_state = nnx.State(restored_dict)
    nnx.update(model, restored_state)
    print(f"Model state loaded from {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Flax NNX VAE on MNIST")
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'],
                        help="Operating mode (default: train)")
    parser.add_argument('--model', type=str, default='model.msgpack',
                        help="File name to save model to or load model from (default: model.msgpack)")
    parser.add_argument('--samples', type=str, default='samples.png',
                        help="File name to save samples in (default: samples.png)")
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help="Batch size for training (default: 128)")
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help="Number of epochs to train (default: 2)")
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
                        help="Dimension of latent variable (default: 10)")
    parser.add_argument('--hidden-dim', type=int, default=512, metavar='N',
                        help="Hidden layer dimension (default: 512)")
    parser.add_argument('--prior', type=str, default='gauss', choices=['gauss', 'mog'],
                        help="Type of prior distribution (default: gauss)")
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help="Random seed (default: 0)")
    parser.add_argument('--n-samples', type=int, default=64, metavar='N',
                        help="Number of samples to generate in sample mode (default: 64)")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help="Learning rate (default: 1e-3)")

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    model_path = os.path.join('models', args.model)
    samples_path = os.path.join('samples', args.samples)

    key = jr.PRNGKey(args.seed)
    key, model_key = jr.split(key)
    rngs = nnx.Rngs(params=model_key)

    img_shape = (28, 28)
    model = init_model(rngs, img_shape, args.latent_dim, args.hidden_dim, args.prior)
    print("Model built successfully.\n")

    if args.mode == 'train':
        mnist_dataset = get_mnist_dataset(
            train_batch_size=args.batch_size,
            num_epochs=args.epochs,
            seed=args.seed,
            train_worker_count=4
        )
        optimizer = nnx.Optimizer(model, optax.adam(args.lr), wrt=nnx.Param)

        print('Training started')
        train(model, optimizer, mnist_dataset, args.epochs, args.batch_size, key)
        print('Training completed')

        save_checkpoint(model, model_path)

    elif args.mode == 'sample':
        load_checkpoint(model, model_path)

        samples_dir = os.path.dirname(samples_path)
        if samples_dir:
            os.makedirs(samples_dir, exist_ok=True)

        key, sample_key = jr.split(key)
        sample(model, args.n_samples, sample_key, samples_path)
        print(f"Samples generated and saved to {samples_path}")


if __name__ == "__main__":
    main()
