import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


NUM_PATHS = 1_000_000


def simulate_path(key):
    """
    Simulates one stochastic business path.

    Random variables:
    D: Market demand, normal distribution
    C: Production asset cost, log-normal distribution
    R: Regulatory penalty rate, uniform distribution
    """
    key_d, key_c, key_r = jax.random.split(key, 3)

    demand = jax.random.normal(key_d) * 150.0 + 1000.0

    log_cost = jax.random.normal(key_c) * 0.3 + 5.5
    cost = jnp.exp(log_cost)

    penalty_rate = jax.random.uniform(
        key_r,
        minval=0.05,
        maxval=0.25
    )

    revenue = (demand * 150.0) - cost * (1.0 - penalty_rate)

    return revenue


@jax.jit
def run_monte_carlo(keys):
    """
    Vectorized Monte Carlo simulation.
    No Python loop is used for the individual simulation paths.
    """
    revenues = jax.vmap(simulate_path)(keys)
    return revenues


def create_histogram(revenues, expected_revenue, var_95):
    os.makedirs("data", exist_ok=True)

    revenues_np = np.array(revenues)

    plt.figure(figsize=(10, 6))
    plt.hist(revenues_np, bins=100, alpha=0.75)

    plt.axvline(
        expected_revenue,
        color="black",
        linewidth=2,
        label=f"Expected Revenue = {expected_revenue:.2f}"
    )

    plt.axvline(
        var_95,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"VaR 95% = {var_95:.2f}"
    )

    plt.title("JAX Monte Carlo Revenue Distribution")
    plt.xlabel("Revenue")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("data/revenue_dist.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)

    print("JAX devices:", jax.devices())
    print("Generating 1,000,000 unique subkeys...")

    keys = jax.random.split(key, NUM_PATHS)

    print("Running JAX Monte Carlo simulation...")

    # First run triggers JIT compilation.
    warmup_start = time.perf_counter()
    revenues = run_monte_carlo(keys)
    revenues.block_until_ready()
    warmup_end = time.perf_counter()

    warmup_time = warmup_end - warmup_start

    # Second run measures warm execution.
    run_start = time.perf_counter()
    revenues = run_monte_carlo(keys)
    revenues.block_until_ready()
    run_end = time.perf_counter()

    execution_time = run_end - run_start

    expected_revenue = jnp.mean(revenues)
    var_95 = jnp.percentile(revenues, 5)

    print("Monte Carlo simulation completed.")
    print("Number of paths:", NUM_PATHS)
    print("JAX warm-up / compilation time:", float(warmup_time), "seconds")
    print("JAX second run execution time:", float(execution_time), "seconds")
    print("Expected revenue:", float(expected_revenue))
    print("Value-at-Risk 95% threshold:", float(var_95))

    create_histogram(revenues, float(expected_revenue), float(var_95))

    print("Saved histogram to data/revenue_dist.png")