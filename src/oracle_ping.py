import os
from google import genai


def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY wurde nicht gefunden. "
            "Setze ihn vorher in PowerShell mit: "
            '$env:GEMINI_API_KEY="DEIN_API_KEY"'
        )

    client = genai.Client(api_key=api_key)

    prompt = (
        "Explain the difference between a stateful NumPy random generation process "
        "and a stateless JAX PRNG split operation in exactly one highly sarcastic sentence."
    )

    models_to_try = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            print(f"Versuche Modell: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            print("Oracle Ping erfolgreich.")
            print("Verwendetes Modell:", model_name)
            print("Antwort:")
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