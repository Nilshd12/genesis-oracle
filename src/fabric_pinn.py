import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from pinn_data import generate_pinn_data


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

    This scalar form is useful because jax.grad needs scalar-input
    functions when differentiating with respect to x or t.
    """
    point = jnp.array([[x, t]])
    u = model.apply(params, point)
    return u[0, 0]


def heat_residual(params, model, x, t, alpha):
    """
    Compute the PDE residual of the 1D heat equation:

        u_t - alpha * u_xx = 0

    using automatic differentiation.
    """

    # First derivative with respect to time t
    u_t = jax.grad(lambda time: predict_u(params, model, x, time))(t)

    # First derivative with respect to space x
    u_x = jax.grad(lambda space: predict_u(params, model, space, t))(x)

    # Second derivative with respect to space x
    u_xx = jax.grad(
        lambda space: jax.grad(
            lambda inner_space: predict_u(params, model, inner_space, t)
        )(space)
    )(x)

    residual = u_t - alpha * u_xx
    return residual


def physics_loss(params, model, pde_points, alpha):
    """
    PDE residual loss evaluated at all collocation points.

    pde_points has shape (N, 2), columns are [x, t].
    """

    def residual_at_point(point):
        x = point[0]
        t = point[1]
        return heat_residual(params, model, x, t, alpha)

    residuals = jax.vmap(residual_at_point)(pde_points)
    return jnp.mean(residuals ** 2)


def ic_loss(params, model, ic_points, u_ic):
    """
    Initial condition loss.

    The PINN must satisfy:
        u(x, 0) = -sin(pi * x)
    """
    predictions = model.apply(params, ic_points)
    return jnp.mean((predictions - u_ic) ** 2)


def bc_loss(params, model, bc_points, u_bc):
    """
    Boundary condition loss.

    The PINN must satisfy:
        u(0, t) = 0
        u(1, t) = 0
    """
    predictions = model.apply(params, bc_points)
    return jnp.mean((predictions - u_bc) ** 2)


def total_loss(params, model, data, alpha):
    """
    Full PINN loss:

        Total_Loss = Physics_Loss + IC_Loss + BC_Loss
    """
    loss_physics = physics_loss(params, model, data["pde_points"], alpha)
    loss_ic = ic_loss(params, model, data["ic_points"], data["u_ic"])
    loss_bc = bc_loss(params, model, data["bc_points"], data["u_bc"])

    return loss_physics + loss_ic + loss_bc


def create_train_step(model, optimizer, data, alpha):
    """
    Creates a JIT-compiled training step.

    This is the basic Optax setup required for training the PINN.
    """

    @jax.jit
    def train_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(total_loss)(
            params,
            model,
            data,
            alpha
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_value

    return train_step


if __name__ == "__main__":
    alpha = 0.1

    key = jax.random.PRNGKey(42)
    data = generate_pinn_data(seed=42)

    model = HeatSurrogate()

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

    print("\nInitial losses:")
    print("Physics loss:", physics_loss(params, model, data["pde_points"], alpha))
    print("IC loss:", ic_loss(params, model, data["ic_points"], data["u_ic"]))
    print("BC loss:", bc_loss(params, model, data["bc_points"], data["u_bc"]))
    print("Total loss:", total_loss(params, model, data, alpha))

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    train_step = create_train_step(model, optimizer, data, alpha)

    params, opt_state, loss_value = train_step(params, opt_state)

    print("\nAfter one training step:")
    print("Total loss:", loss_value)