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


def split_data(signal, periods=100, split_period=60):
    samples_per_period = len(signal) // periods
    split_index = split_period * samples_per_period

    train_signal = signal[:split_index]
    test_signal = signal[split_index:]

    return train_signal, test_signal


class SignalCompression(layers.Layer):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.conv1 = layers.Conv1D(filters=16, kernel_size=3, strides=2, activation="relu", padding="same")
        self.conv2 = layers.Conv1D(filters=32, kernel_size=3, strides=1, activation="relu", padding="same")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(self.latent_dim, activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)


class SignalExpansion(layers.Layer):
    def __init__(self, output_dim=50):
        super().__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.seq_len = (self.output_dim + 1) // 2 
        self.dense = layers.Dense(self.seq_len * 32, activation="relu")
        self.reshape = layers.Reshape((self.seq_len, 32))
        self.deconv1 = layers.Conv1DTranspose(filters=16, kernel_size=3, strides=1, activation="relu", padding="same")
        self.deconv2 = layers.Conv1DTranspose(filters=1, kernel_size=3, strides=2, activation="linear", padding="same")
        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        
        x = x[:, :self.output_dim, :]
        return self.flatten(x)


class PhysicsAutoencoder(keras.Model):
    def __init__(self, window_size=50, latent_dim=8):
        super().__init__()
        self.window_size = window_size
        self.encoder = SignalCompression(latent_dim)
        self.decoder = SignalExpansion(window_size)
        self.expand_dims = layers.Reshape((window_size, 1))

    def call(self, inputs):
        x = self.expand_dims(inputs)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
# dient nur als test
if __name__ == "__main__":
    dummy = np.random.rand(10, 50)
    
    model = PhysicsAutoencoder()
    output = model(dummy)
    
    print("Input shape:", dummy.shape)
    print("Output shape:", output.shape)