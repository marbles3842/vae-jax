# Bernoulli VAE with Flax NNX

This project is a clean, minimal implementation of a Bernoulli Variational Autoencoder (VAE) using JAX and the newer Flax NNX API. It trains a VAE on the MNIST dataset using ancestral sampling to generate new binary digits.

## Project Structure

*   `vae.py`: Defines the VAE model components (prior, encoder, decoder, and ELBO calculation) as Flax NNX modules using Distrax for probability distributions.
*   `mnist_data_loader.py`: Uses Google's Grain and Hugging Face Datasets to load, binarize, and batch the MNIST dataset.
*   `train.py`: The training loop which initializes the model and optimizer, trains the VAE for 5 epochs using Optax, and saves generated digit samples.
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

## How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the model**:
    ```bash
    python train.py
    ```
    This will load MNIST, train the VAE, and output `samples.png` containing generated digits.
