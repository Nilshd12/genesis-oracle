import time
import numpy as np


def simulate_legacy_swarm(num_oscillators=100_000, num_steps=1_000, dt=0.01, damping=0.05):
    rng = np.random.default_rng(42)

    # Slightly different natural frequencies for each oscillator
    w = rng.uniform(0.5, 2.0, size=num_oscillators)

    # Initial positions and velocities
    x = np.ones(num_oscillators)
    v = np.zeros(num_oscillators)

    for _ in range(num_steps):
        # Damped harmonic oscillator:
        # x'' = -damping * x' - w^2 * x
        a = -damping * v - (w ** 2) * x

        # Explicit Euler integration
        v = v + dt * a
        x = x + dt * v

    return x, v


if __name__ == "__main__":
    start = time.time()
    final_x, final_v = simulate_legacy_swarm()
    end = time.time()

    elapsed = end - start

    print("Legacy simulation completed.")
    print("Number of oscillators:", len(final_x))
    print("Execution time:", elapsed, "seconds")

# hat 1.1252140998840332 seconds zum ausführen gebraucht.