import jax
import jax.numpy as jnp


def generate_pinn_data(seed=42, n_collocation=5000, n_ic=500, n_bc=500):
    """
    Generates mesh-free training points for a PINN solving the 1D heat equation.

    Domain:
        x in [0, 1]
        t in [0, 1]

    Datasets:
        - Collocation points for the PDE residual
        - Initial condition points at t = 0
        - Boundary condition points at x = 0 and x = 1
    """

    key = jax.random.PRNGKey(seed)
    key_pde, key_ic, key_bc = jax.random.split(key, 3)

    # ------------------------------------------------------------
    # 1. Collocation points for PDE residual
    # Random points inside the full space-time domain.
    # Shape: (n_collocation, 2), columns are [x, t]
    # ------------------------------------------------------------
    pde_points = jax.random.uniform(
        key_pde,
        shape=(n_collocation, 2),
        minval=0.0,
        maxval=1.0
    )

    # ------------------------------------------------------------
    # 2. Initial condition points
    # x is random in [0, 1], t is fixed to 0.
    # u(x, 0) = -sin(pi * x)
    # ------------------------------------------------------------
    x_ic = jax.random.uniform(
        key_ic,
        shape=(n_ic, 1),
        minval=0.0,
        maxval=1.0
    )

    t_ic = jnp.zeros_like(x_ic)
    u_ic = -jnp.sin(jnp.pi * x_ic)

    ic_points = jnp.concatenate([x_ic, t_ic], axis=1)

    # ------------------------------------------------------------
    # 3. Boundary condition points
    # Half of the points are at x = 0, half at x = 1.
    # t is random in [0, 1].
    # u(0, t) = 0 and u(1, t) = 0
    # ------------------------------------------------------------
    n_left = n_bc // 2
    n_right = n_bc - n_left

    t_left = jax.random.uniform(
        key_bc,
        shape=(n_left, 1),
        minval=0.0,
        maxval=1.0
    )

    key_bc_2 = jax.random.fold_in(key_bc, 1)

    t_right = jax.random.uniform(
        key_bc_2,
        shape=(n_right, 1),
        minval=0.0,
        maxval=1.0
    )

    x_left = jnp.zeros_like(t_left)
    x_right = jnp.ones_like(t_right)

    left_boundary = jnp.concatenate([x_left, t_left], axis=1)
    right_boundary = jnp.concatenate([x_right, t_right], axis=1)

    bc_points = jnp.concatenate([left_boundary, right_boundary], axis=0)
    u_bc = jnp.zeros((n_bc, 1))

    return {
        "pde_points": pde_points,
        "ic_points": ic_points,
        "u_ic": u_ic,
        "bc_points": bc_points,
        "u_bc": u_bc,
    }


if __name__ == "__main__":
    data = generate_pinn_data()

    print("PINN data generated successfully.")
    print("PDE collocation points:", data["pde_points"].shape)
    print("Initial condition points:", data["ic_points"].shape)
    print("Initial condition values:", data["u_ic"].shape)
    print("Boundary condition points:", data["bc_points"].shape)
    print("Boundary condition values:", data["u_bc"].shape)

    print("\nExample IC point [x, t] and u:")
    print(data["ic_points"][0], data["u_ic"][0])

    print("\nExample BC point [x, t] and u:")
    print(data["bc_points"][0], data["u_bc"][0])