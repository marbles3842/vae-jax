# Bernoulli VAE with Flax NNX

This project is a clean, minimal implementation of a Bernoulli Variational Autoencoder (VAE) using JAX and the Flax NNX API. It trains a VAE on the MNIST dataset using ancestral sampling to generate new binary digits.

## Project Structure

*   `vae_jax/models/vae.py`: Defines the VAE model components (prior, encoder, decoder, and ELBO calculation) as Flax NNX modules using Distrax for probability distributions.
*   `vae_jax/data/mnist.py`: Uses Google's Grain and Hugging Face Datasets to load, binarize, and batch the MNIST dataset.
*   `main.py`: The central execution entry point which parses arguments, manages model saving/loading, and runs the training or sampling pipeline.
*   `train.py`: Contains training loop logic, including `train_step`, and updates an interactive progress bar using `tqdm`.
*   `sample.py`: Contains VAE sample generation logic and grid image saving function (`save_samples`).
*   `requirements.txt`: Specifies the core Python package requirements.

## Dependencies

The implementation relies on the following key dependencies:
*   **JAX / Jaxlib**: Core high-performance numerical computing library.
*   **Flax (NNX)**: Neural network library for JAX (utilizing NNX's module system).
*   **Distrax**: Probability distributions library for JAX.
*   **Optax**: Gradient processing and optimization library.
*   **Grain**: Library for fast data loading and preprocessing.
*   **Hugging Face Datasets**: Used for fetching the raw MNIST dataset.
*   **Matplotlib & NumPy**: For data manipulation and saving generated samples.
*   **tqdm**: For displaying the interactive progress bar.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
To train the VAE on MNIST:
```bash
python main.py train --epochs 2 --batch-size 128 --model model.msgpack
```
This trains the model for 2 epochs and saves the model parameters to `models/model.msgpack`.

### 3. Generate samples
To load the trained model parameters and generate digit samples:
```bash
python main.py sample --model model.msgpack --samples samples.png --n-samples 64
```
This loads parameters from `models/model.msgpack` and generates `samples/samples.png` containing the generated digits.
