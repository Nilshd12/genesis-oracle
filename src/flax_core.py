import jax
import jax.numpy as jnp
from flax import linen as nn

class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP)."""
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        # Hidden layer with ReLU activation
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        # Output layer
        x = nn.Dense(features=self.out_dim)(x)
        return x

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # EXPLICIT STATE MANAGEMENT IN FLAX
    # -------------------------------------------------------------------------
    # In Keras, models are stateful. When you instantiate a Keras model like
    # `model = Sequential(...)`, the weights are implicitly created and stored 
    # inside the `model` object itself.
    # 
    # In Flax, models are strictly STATELESS. The `SimpleMLP` object only 
    # describes the *architecture* (the math operations), but it does NOT 
    # contain or store any weights. 
    # 
    # Let's see how this works in practice:

    # 1. Instantiate the model architecture. 
    #    (This object has NO weights attached to it!)
    model = SimpleMLP(hidden_dim=32, out_dim=10)

    # 2. Create a pseudo-random number generator (PRNG) key.
    #    JAX requires explicit random state passing for reproducible randomness.
    rng_key = jax.random.PRNGKey(seed=42)

    # 3. Create some dummy input data.
    #    We need this because Flax shapes the weights based on the input size.
    dummy_input = jnp.ones((1, 16))  # e.g., batch size 1, 16 input features

    # 4. INITIALIZATION: Explicitly generate the weights.
    #    We pass the PRNG key and dummy input to `model.init()`.
    #    This returns a dictionary (specifically, a FrozenDict) containing 
    #    all the initialized parameters (weights and biases).
    print("--- Initializing Parameters ---")
    variables = model.init(rng_key, dummy_input)
    
    # Let's look at the structure of our explicitly generated parameters:
    # They are completely separate from the `model` object.
    params = variables['params']
    print("Parameter structure:")
    print(jax.tree_util.tree_map(lambda x: x.shape, params))
    
    # 5. FORWARD PASS: Explicitly pass the parameters into the model.
    #    Unlike Keras where you do `output = model(input)`, in Flax we must
    #    explicitly provide BOTH the weights AND the input to `model.apply()`.
    print("\n--- Running Forward Pass ---")
    # Generate some real random input for the forward pass
    input_data = jax.random.normal(rng_key, (1, 16))
    
    output = model.apply(variables, input_data)
    
    print("Input shape: ", input_data.shape)
    print("Output shape:", output.shape)
    print("Output data: \n", output)
