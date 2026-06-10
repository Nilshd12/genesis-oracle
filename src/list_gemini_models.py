import os
from google import genai


def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY wurde nicht gefunden. "
            "Setze ihn vorher mit: "
            '$env:GEMINI_API_KEY="DEIN_API_KEY"'
        )

    client = genai.Client(api_key=api_key)

    print("Verfügbare Modelle:")
    print("-" * 60)

    for model in client.models.list():
        print(model.name)

        supported_actions = getattr(model, "supported_actions", None)
        if supported_actions:
            print("  supported_actions:", supported_actions)

        supported_generation_methods = getattr(
            model,
            "supported_generation_methods",
            None
        )
        if supported_generation_methods:
            print("  supported_generation_methods:", supported_generation_methods)

        print()


if __name__ == "__main__":
    main()