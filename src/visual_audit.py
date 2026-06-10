import os
from pathlib import Path

from google import genai
from google.genai import types


def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY wurde nicht gefunden. "
            "Setze ihn vorher in PowerShell mit: "
            '$env:GEMINI_API_KEY="DEIN_API_KEY"'
        )

    image_path = Path("data/audit_target.png")

    if not image_path.exists():
        raise FileNotFoundError(
            "data/audit_target.png wurde nicht gefunden. "
            "Führe zuerst aus: uv run python src/generate_signals.py"
        )

    client = genai.Client(api_key=api_key)

    prompt = (
        "You are a Visual Detective auditing a generated engineering signal plot. "
        "Inspect the image carefully. Find the visual anomaly or malfunction in the waveform. "
        "Estimate the approximate X-axis region where the malfunction occurs. "
        "Then write a short funny poem mocking the engineering team that allowed this bug to pass. "
        "Keep the answer concise."
    )

    image_bytes = image_path.read_bytes()

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-flash-lite-latest",
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            print(f"Versuche Modell: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    ),
                    prompt,
                ],
            )

            print("Visual Audit erfolgreich.")
            print("Verwendetes Modell:", model_name)
            print("\nDiagnose:")
            print(response.text)
            return

        except Exception as error:
            print(f"Fehler mit {model_name}: {error}")
            last_error = error

    raise RuntimeError(
        f"Alle Modelle sind fehlgeschlagen. Letzter Fehler: {last_error}"
    )


if __name__ == "__main__":
    main()