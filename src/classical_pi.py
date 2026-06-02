import os
import time
import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(num_points=1_000_000, sample_size=5_000):
    """
    Classical NumPy Monte Carlo estimation of pi using stateful randomness.

    Points are sampled uniformly in the unit square [0, 1] x [0, 1].
    The ratio of points inside the quarter circle estimates pi / 4.
    """
    start_time = time.perf_counter()

    x = np.random.uniform(0.0, 1.0, size=num_points)
    y = np.random.uniform(0.0, 1.0, size=num_points)

    distances_squared = x ** 2 + y ** 2
    inside = distances_squared <= 1.0

    pi_estimate = 4.0 * np.mean(inside)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Random subset for visualization so the plot stays readable.
    subset_indices = np.random.choice(num_points, size=sample_size, replace=False)

    plot_x = x[subset_indices]
    plot_y = y[subset_indices]
    plot_inside = inside[subset_indices]

    return pi_estimate, execution_time, plot_x, plot_y, plot_inside


def create_plot(plot_x, plot_y, plot_inside, pi_estimate):
    os.makedirs("data", exist_ok=True)

    theta = np.linspace(0.0, np.pi / 2.0, 300)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        plot_x[plot_inside],
        plot_y[plot_inside],
        s=5,
        alpha=0.6,
        label="Inside circle"
    )

    plt.scatter(
        plot_x[~plot_inside],
        plot_y[~plot_inside],
        s=5,
        alpha=0.6,
        label="Outside circle"
    )

    plt.plot(circle_x, circle_y, linewidth=2, label="Quarter-circle boundary")

    plt.title(f"Classical Monte Carlo Pi Estimation: pi ≈ {pi_estimate:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("data/classical_pi_disp.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    pi_estimate, execution_time, plot_x, plot_y, plot_inside = estimate_pi()

    print("Classical NumPy Pi estimation completed.")
    print("Estimated pi:", pi_estimate)
    print("Execution time:", execution_time, "seconds")

    create_plot(plot_x, plot_y, plot_inside, pi_estimate)
    print("Saved plot to data/classical_pi_disp.png")