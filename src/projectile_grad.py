import os

os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp


def projectile_loss(v_initial):
    """
    Simulates the final horizontal distance of a projectile after 5 seconds
    and returns the MSE to the target distance of 150.0 meters.
    """
    target_distance = 150.0
    flight_time = 5.0

    simulated_distance = v_initial * flight_time

    loss = (simulated_distance - target_distance) ** 2
    return loss


def optimize_initial_velocity(start_velocity=10.0, learning_rate=0.01, iterations=20):
    grad_fn = jax.grad(projectile_loss)

    v = start_velocity

    for step in range(iterations):
        gradient = grad_fn(v)
        v = v - learning_rate * gradient

        print(
            f"Step {step + 1:02d}: "
            f"v_initial = {float(v):.6f}, "
            f"loss = {float(projectile_loss(v)):.6f}, "
            f"gradient = {float(gradient):.6f}"
        )

    return v


if __name__ == "__main__":
    optimized_velocity = optimize_initial_velocity()

    print()
    print("Optimized initial velocity:", float(optimized_velocity), "m/s")