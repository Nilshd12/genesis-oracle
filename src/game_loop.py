import json
import os
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from sandbox_env import simulate_thermal_dampener, format_temperature_log


class ControlDecision(BaseModel):
    system_state: Literal["FREEZING", "BOILING", "PERFECT"] = Field(
        description="Must be 'FREEZING', 'BOILING', or 'PERFECT'"
    )
    adjustment_action: Literal["INCREASE", "DECREASE", "HOLD"] = Field(
        description="Must be 'INCREASE', 'DECREASE', or 'HOLD'"
    )
    delta_value: float = Field(
        description="The exact numerical change to apply to Kappa"
    )
    confidence_score: float = Field(
        description="Confidence score between 0.0 and 1.0"
    )


def get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY wurde nicht gefunden. "
            "Setze ihn vorher in PowerShell mit: "
            '$env:GEMINI_API_KEY="DEIN_API_KEY"'
        )

    return api_key


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def apply_decision(kappa: float, decision: ControlDecision) -> float:
    delta = abs(decision.delta_value)

    if decision.adjustment_action == "INCREASE":
        kappa += delta
    elif decision.adjustment_action == "DECREASE":
        kappa -= delta
    elif decision.adjustment_action == "HOLD":
        kappa = kappa

    return clamp(kappa, 0.0, 2.0)


def parse_decision_response(response) -> ControlDecision:
    if hasattr(response, "parsed") and response.parsed is not None:
        return response.parsed

    response_text = response.text.strip()
    data = json.loads(response_text)
    return ControlDecision.model_validate(data)


def request_control_decision(client, model_name: str, temperature_log: str) -> ControlDecision:
    prompt = f"""
You are an autonomous thermal control regulator.

Your task:
Analyze the thermal dampener log and return exactly one structured JSON object following the schema.

Rules:
- If the system is FREEZING, choose INCREASE.
- If the system is BOILING, choose DECREASE.
- If the system is PERFECT, choose HOLD.
- Kappa should move toward the stable target around 1.0.
- Use a delta_value between 0.05 and 0.40.
- Use HOLD with delta_value 0.0 only if the system is PERFECT.
- Do not include markdown.
- Do not include explanations outside JSON.

Current telemetry:
{temperature_log}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ControlDecision,
            temperature=0.1,
        ),
    )

    return parse_decision_response(response)


def main():
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-flash-lite-latest",
    ]

    model_name = models_to_try[0]

    kappa = 0.2

    print("=== Cerebral Nexus Control Loop ===")
    print(f"Start-Kappa: {kappa:.4f}")
    print(f"Verwendetes Modell: {model_name}")
    print()

    for turn in range(1, 6):
        print("=" * 80)
        print(f"TURN {turn}")

        result = simulate_thermal_dampener(kappa)
        temperature_log = format_temperature_log(result)

        print("\nTelemetry Input:")
        print(temperature_log)

        try:
            decision = request_control_decision(
                client=client,
                model_name=model_name,
                temperature_log=temperature_log,
            )
        except Exception as error:
            print(f"Fehler mit Modell {model_name}: {error}")
            raise

        print("\nStructured JSON Decision:")
        print(decision.model_dump_json(indent=2))

        old_kappa = kappa
        kappa = apply_decision(kappa, decision)

        print("\nApplied Control:")
        print(f"Old Kappa: {old_kappa:.4f}")
        print(f"Action: {decision.adjustment_action}")
        print(f"Delta: {decision.delta_value:.4f}")
        print(f"New Kappa: {kappa:.4f}")

        updated_result = simulate_thermal_dampener(kappa)

        print("\nUpdated System State:")
        print(f"Average temperature: {updated_result.average_temperature:.2f} K")
        print(f"System state: {updated_result.system_state}")

        if updated_result.system_state == "PERFECT":
            print("\nPERFECT zone reached.")
            break

    print("\n=== Final Result ===")
    final_result = simulate_thermal_dampener(kappa)
    print(format_temperature_log(final_result))


if __name__ == "__main__":
    main()