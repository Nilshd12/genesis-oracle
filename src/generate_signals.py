import os
import numpy as np
import matplotlib.pyplot as plt


def generate_signal(num_points=1000, seed=42):
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 10, num_points)

    # Dynamisches Wellensignal aus mehreren Frequenzen
    signal = (
        np.sin(2 * np.pi * 0.7 * t)
        + 0.4 * np.sin(2 * np.pi * 2.5 * t)
        + 0.15 * rng.normal(size=num_points)
    )

    # Geheimer Fehler: zufälliger Bereich mit hochfrequentem Clipping-Artefakt
    artifact_start = rng.integers(low=250, high=750)
    artifact_width = 80
    artifact_end = artifact_start + artifact_width

    high_freq_artifact = 2.5 * np.sin(
        2 * np.pi * 18.0 * t[artifact_start:artifact_end]
    )

    signal[artifact_start:artifact_end] += high_freq_artifact

    # Clipping / Amplitudensättigung
    signal[artifact_start:artifact_end] = np.clip(
        signal[artifact_start:artifact_end],
        -1.2,
        1.2
    )

    return t, signal


def save_plot(t, signal, output_path="data/audit_target.png"):
    os.makedirs("data", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(t, signal, linewidth=1.5)
    plt.title("Dynamic Wave Signal for Visual Audit")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    t, signal = generate_signal()
    save_plot(t, signal)

    print("Signal plot saved to data/audit_target.png")