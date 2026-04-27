import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import random

print("Keras backend:", keras.backend.backend())

x = random.normal(shape=(2, 3))
print("Random tensor:")
print(x)
print("Tensor type:", type(x))