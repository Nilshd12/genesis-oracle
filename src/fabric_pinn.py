import jax
import jax.numpy as jnp
from flax import linen as nn


class HeatSurrogate(nn.Module):
    """
    Flax MLP surrogate for a Physics-Informed Neural Network.

    Input:
        points with shape (..., 2), where each point is [x, t]

    Output:
        predicted temperature u with shape (..., 1)
    """
    hidden_dim: int = 32
    num_hidden_layers: int = 4
    output_dim: int = 1

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.tanh(x)

        x = nn.Dense(self.output_dim)(x)
        return x


def predict_u(params, model, x, t):
    """
    Predict scalar temperature u(x, t).

    x and t are scalar values.
    This function is intentionally scalar-friendly because later
    jax.grad will differentiate with respect to x and t.
    """
    point = jnp.array([[x, t]])
    u = model.apply(params, point)
    return u[0, 0]


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    model = HeatSurrogate()

    # Example batch of 3 continuous space-time points:
    # [x, t]
    sample_points = jnp.array([
        [0.0, 0.0],
        [0.5, 0.25],
        [1.0, 1.0],
    ])

    params = model.init(key, sample_points)

    output = model.apply(params, sample_points)

    print("HeatSurrogate initialized successfully.")
    print("Input shape:", sample_points.shape)
    print("Output shape:", output.shape)
    print("Output:")
    print(output)

    print("\nParameter structure:")
    print(jax.tree_util.tree_map(lambda value: value.shape, params))