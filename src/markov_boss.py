import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


NUM_DAYS = 365


def get_transition_matrix(day):
    """
    Returns the transition matrix for a given day.

    Baseline states:
    State 0: Bull Market
    State 1: Stagnation
    State 2: Catastrophic Recession

    During the Black Swan shock from day 180 to day 190,
    State 0 and State 1 are forced to shift 80% probability mass
    directly into State 2.
    """

    baseline = jnp.array([
        [0.85, 0.12, 0.03],
        [0.10, 0.75, 0.15],
        [0.05, 0.20, 0.75],
    ])

    shock = jnp.array([
        [0.10, 0.10, 0.80],
        [0.10, 0.10, 0.80],
        [0.05, 0.20, 0.75],
    ])

    is_shock_day = (day >= 180) & (day < 190)

    return jnp.where(is_shock_day, shock, baseline)


@jax.jit
def simulate_markov_chain(num_days=NUM_DAYS):
    """
    Simulates the aggregate Markov probability vector over time
    using Module Alpha: Matrix Carrier.

    The initial state assumes that the enterprise starts in a Bull Market.
    """

    initial_state = jnp.array([1.0, 0.0, 0.0])

    def step(state_vector, day):
        transition_matrix = get_transition_matrix(day)
        next_state = state_vector @ transition_matrix
        return next_state, next_state

    days = jnp.arange(num_days)

    final_state, state_history = jax.lax.scan(
        step,
        initial_state,
        days
    )

    # Add day 0 initial state at the beginning for plotting.
    state_history = jnp.vstack([initial_state, state_history])

    return state_history


def create_plot(state_history):
    os.makedirs("data", exist_ok=True)

    state_history_np = np.array(state_history) * 100.0
    days = np.arange(state_history_np.shape[0])

    plt.figure(figsize=(12, 6))

    plt.plot(days, state_history_np[:, 0], label="State 0: Bull Market")
    plt.plot(days, state_history_np[:, 1], label="State 1: Stagnation")
    plt.plot(days, state_history_np[:, 2], label="State 2: Catastrophic Recession")

    plt.axvspan(180, 190, alpha=0.2, label="Black Swan Shock")

    plt.title("Macro-Economic Markov Chain with Black Swan Shock")
    plt.xlabel("Day")
    plt.ylabel("State Distribution (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("data/markov_boss.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Running Markov Boss simulation...")
    print("Chosen strategy: Module Alpha (The Matrix Carrier)")
    print("Tracking aggregate probability state vector over 365 days.")

    state_history = simulate_markov_chain()
    state_history.block_until_ready()

    create_plot(state_history)

    final_distribution = np.array(state_history[-1]) * 100.0
    shock_distribution = np.array(state_history[190]) * 100.0

    print("\nDistribution at day 190 after Black Swan shock:")
    print(f"Bull Market: {shock_distribution[0]:.2f}%")
    print(f"Stagnation: {shock_distribution[1]:.2f}%")
    print(f"Catastrophic Recession: {shock_distribution[2]:.2f}%")

    print("\nFinal distribution at day 365:")
    print(f"Bull Market: {final_distribution[0]:.2f}%")
    print(f"Stagnation: {final_distribution[1]:.2f}%")
    print(f"Catastrophic Recession: {final_distribution[2]:.2f}%")

    print("\nSaved plot to data/markov_boss.png")