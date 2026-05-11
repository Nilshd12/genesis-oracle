import time

import jax
import jax.numpy as jnp


def oscillator_step(x, v, w, dt=0.01, damping=0.05):
    """
    One explicit Euler step for a single damped harmonic oscillator.

    x: position
    v: velocity
    w: natural frequency
    """
    a = -damping * v - (w ** 2) * x
    new_v = v + dt * a
    new_x = x + dt * new_v
    return new_x, new_v


# Vectorize the single oscillator step over many oscillators.
vectorized_step = jax.vmap(
    oscillator_step,
    in_axes=(0, 0, 0, None, None)
)


@jax.jit
def simulate_jax_swarm(w, num_steps=1000, dt=0.01, damping=0.05):
    """
    JIT-compiled JAX simulation for many oscillators.
    Uses lax.scan as a JAX-native loop over time steps.
    """
    x = jnp.ones_like(w)
    v = jnp.zeros_like(w)

    def scan_body(state, _):
        x, v = state
        x, v = vectorized_step(x, v, w, dt, damping)
        return (x, v), None

    (final_x, final_v), _ = jax.lax.scan(
        scan_body,
        (x, v),
        xs=None,
        length=num_steps
    )

    return final_x, final_v


if __name__ == "__main__":
    num_oscillators = 100_000
    num_steps = 1_000

    key = jax.random.PRNGKey(42)
    w = jax.random.uniform(
        key,
        shape=(num_oscillators,),
        minval=0.5,
        maxval=2.0
    )

    print("Running JAX simulation...")
    print("Number of oscillators:", num_oscillators)
    print("Number of time steps:", num_steps)
    print("JAX devices:", jax.devices())

    # First run: warm-up / tracing / compilation
    start = time.time()
    final_x, final_v = simulate_jax_swarm(w, num_steps=num_steps)
    final_x.block_until_ready()
    end = time.time()
    warmup_time = end - start

    print("Warm-up / compilation time:", warmup_time, "seconds")

    # Second run: actual measured execution time
    start = time.time()
    final_x, final_v = simulate_jax_swarm(w, num_steps=num_steps)
    final_x.block_until_ready()
    end = time.time()
    second_run_time = end - start

    print("JAX second run time:", second_run_time, "seconds")