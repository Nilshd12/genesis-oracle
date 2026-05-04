import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from keras import layers


def create_windows(signal, window_size=50):
    windows = []

    for i in range(len(signal) - window_size + 1):
        windows.append(signal[i:i + window_size])

    return np.array(windows)


def split_data(signal, T, split_period=60):
    split_index = int(split_period * T)

    train_signal = signal[:split_index]
    test_signal = signal[split_index:]

    return train_signal, test_signal


class SignalCompression(layers.Layer):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.dense = layers.Dense(self.latent_dim)

    def call(self, inputs):
        return keras.activations.relu(self.dense(inputs))


class SignalExpansion(layers.Layer):
    def __init__(self, output_dim=50):
        super().__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.dense = layers.Dense(self.output_dim)

    def call(self, inputs):
        return self.dense(inputs)


class PhysicsAutoencoder(keras.Model):
    def __init__(self, window_size=50, latent_dim=8):
        super().__init__()
        self.encoder = SignalCompression(latent_dim)
        self.decoder = SignalExpansion(window_size)

    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed
    
# dient nur als test
if __name__ == "__main__":
    dummy = np.random.rand(10, 50)
    
    model = PhysicsAutoencoder()
    output = model(dummy)
    
    print("Input shape:", dummy.shape)
    print("Output shape:", output.shape)