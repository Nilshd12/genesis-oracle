import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # persoenliche Var:
    suffix_s = 808

    # Automatisch daraus berechnet:
    last_two_digits = suffix_s % 100
    last_three_digits = suffix_s

    T = last_two_digits
    if T == 0:
        raise ValueError("Die letzten zwei Ziffern dürfen nicht 00 sein, da T sonst 0 wäre.")

    R = 0.5e3
    C = (1000 + last_three_digits) * 1e-6

    omega_0 = 2 * np.pi / T

    periods = 100
    samples_per_period = 200
    t = np.linspace(0, periods * T, periods * samples_per_period)

    harmonics = np.arange(1, 18, 2)

    original_signal = np.zeros_like(t)
    filtered_signal = np.zeros_like(t)

    for k in harmonics:
        amplitude = 4 / (np.pi * k)
        omega_k = k * omega_0

        original_signal += amplitude * np.sin(omega_k * t)

        H = 1 / (1 + 1j * omega_k * R * C)
        gain = np.abs(H)
        phase = np.angle(H)

        filtered_signal += amplitude * gain * np.sin(omega_k * t + phase)

    rng = np.random.default_rng(42)
    noisy_signal = filtered_signal + rng.normal(0, 0.08, size=t.shape)

    anomaly_signal = noisy_signal.copy()
    anomaly_mask = (t >= 70 * T) & (t <= 75 * T)

    spike = 2.5 * np.sin(40 * omega_0 * t[anomaly_mask])
    anomaly_signal[anomaly_mask] += spike

    os.makedirs("data", exist_ok=True)
    np.save("data/rc_filter_signal.npy", anomaly_signal)

    normal_mask = (t >= 20 * T) & (t <= 25 * T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t[normal_mask], anomaly_signal[normal_mask])
    axes[0].set_title("Normal noisy RC-filtered signal")
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("Voltage")

    axes[1].plot(t[anomaly_mask], anomaly_signal[anomaly_mask])
    axes[1].set_title("Injected anomaly spike")
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Voltage")

    plt.tight_layout()
    plt.savefig("data/data_feed.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()