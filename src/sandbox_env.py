from dataclasses import dataclass
import numpy as np


@dataclass
class ThermalResult:
    kappa: float
    temperatures: list[float]
    average_temperature: float
    system_state: str


def classify_temperature(average_temperature: float) -> str:
    if average_temperature < 295.0:
        return "FREEZING"

    if average_temperature > 305.0:
        return "BOILING"

    return "PERFECT"


def simulate_thermal_dampener(kappa: float, num_steps: int = 20) -> ThermalResult:
    """
    Simulates a simple thermal dampener.

    Kappa is the internal control parameter:
    - too low  -> system freezes
    - too high -> system boils/explodes
    - near 1.0 -> system is stable/perfect
    """

    time_steps = np.arange(num_steps)

    base_temperature = 300.0
    kappa_effect = (kappa - 1.0) * 70.0

    transient_wave = 4.0 * np.sin(time_steps * 0.7)
    small_noise = 1.5 * np.cos(time_steps * 0.3)

    temperatures = base_temperature + kappa_effect + transient_wave + small_noise

    average_temperature = float(np.mean(temperatures[-5:]))
    system_state = classify_temperature(average_temperature)

    return ThermalResult(
        kappa=float(kappa),
        temperatures=[float(value) for value in temperatures],
        average_temperature=average_temperature,
        system_state=system_state,
    )


def format_temperature_log(result: ThermalResult) -> str:
    rounded_temperatures = [round(value, 2) for value in result.temperatures]

    return (
        f"Kappa: {result.kappa:.4f}\n"
        f"Temperature log: {rounded_temperatures}\n"
        f"Average temperature over last 5 steps: {result.average_temperature:.2f} K\n"
        f"Current system state: {result.system_state}"
    )


if __name__ == "__main__":
    test_kappa = 0.2
    result = simulate_thermal_dampener(test_kappa)
    print(format_temperature_log(result))