from pathlib import Path
import subprocess
import re

from google.adk.agents.llm_agent import Agent
from google.genai import types


# Absoluter Pfad zum Hauptverzeichnis von genesis-oracle
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pfad zum geklonten arXiv-Skill
ARXIV_SKILL_DIRECTORY = (
    PROJECT_ROOT
    / "science-skills"
    / "skills"
    / "literature_search_arxiv"
)


def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Searches arXiv for scientific papers by using the Google DeepMind
    Science-Skills command-line tool.

    Args:
        query: The scientific search query sent to arXiv.
        max_results: The maximum number of papers to return.

    Returns:
        The JSON-formatted output produced by the arXiv search tool.
        If the search cannot be completed, an error message is returned.
    """

    if not query.strip():
        return "Error: The arXiv search query must not be empty."

    if max_results < 1:
        return "Error: max_results must be at least 1."

    # Begrenzung, damit keine unnötig große Tool-Antwort entsteht
    max_results = min(max_results, 3)

    script_path = (
        ARXIV_SKILL_DIRECTORY
        / "scripts"
        / "search_arxiv.py"
    )

    if not script_path.exists():
        return (
            "Error: The arXiv search script was not found at "
            f"{script_path}"
        )

    command = [
        "uv",
        "run",
        "scripts/search_arxiv.py",
        "--query",
        query,
        "--max_results",
        str(max_results),
    ]

    try:
        result = subprocess.run(
            command,
            cwd=ARXIV_SKILL_DIRECTORY,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )

    except subprocess.TimeoutExpired:
        return (
            "Error: The arXiv search timed out after "
            "120 seconds."
        )

    except OSError as error:
        return (
            "Error: The arXiv search process could not "
            f"be started: {error}"
        )

    if result.returncode != 0:
        error_message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "Unknown command-line error."
        )

        return (
            "Error: The arXiv search command failed.\n"
            f"Details: {error_message}"
        )

    output = result.stdout.strip()

    if not output:
        return "Error: The arXiv search returned no output."

    return output

def extract_parameters_from_text(text: str) -> dict[str, object]:
    """
    Extracts physical and simulation-related parameters from scientific text.

    The function searches for common reactor and material parameters, such as
    thermal conductivity, temperature, density, heat capacity, friction
    coefficients, fission barrier heights, and fission isomer energies.

    Args:
        text: Scientific abstract or paper text from which parameters
            should be extracted.

    Returns:
        A dictionary containing detected parameters, numerical values,
        units, evidence snippets, and an extraction summary.
    """

    if not text or not text.strip():
        return {
            "status": "error",
            "message": "The input text must not be empty.",
            "parameters": [],
        }

    parameter_definitions = [
        {
            "name": "thermal_conductivity",
            "label_pattern": r"thermal conductivity",
            "unit_pattern": r"W\s*/?\s*\(?m\s*[·*]?\s*K\)?",
            "default_unit": "W/(m·K)",
        },
        {
            "name": "temperature",
            "label_pattern": r"(?:temperature|core temperature)",
            "unit_pattern": r"(?:K|°C|Celsius|Kelvin)",
            "default_unit": None,
        },
        {
            "name": "density",
            "label_pattern": (
                r"\b(?:material density|mass density|bulk density|fuel density)\b"
            ),
            "unit_pattern": (
                r"(?:kg\s*/\s*m(?:\^?3|³)|g\s*/\s*cm(?:\^?3|³))"
            ),
            "default_unit": None,
        },
        {
            "name": "specific_heat_capacity",
            "label_pattern": r"(?:specific heat capacity|heat capacity)",
            "unit_pattern": r"J\s*/\s*\(?kg\s*[·*]?\s*K\)?",
            "default_unit": "J/(kg·K)",
        },
        {
            "name": "thermal_friction_coefficient",
            "label_pattern": r"(?:thermal friction coefficient|friction coefficient)",
            "unit_pattern": r"(?:dimensionless|1)",
            "default_unit": "dimensionless",
        },
        {
            "name": "fission_barrier_height",
            "label_pattern": r"(?:inner|outer|fission)?\s*barrier heights?",
            "unit_pattern": r"(?:MeV|keV|eV)",
            "default_unit": "MeV",
        },
        {
            "name": "fission_isomer_energy",
            "label_pattern": r"(?:fission )?isomer energies?",
            "unit_pattern": r"(?:MeV|keV|eV)",
            "default_unit": "MeV",
        },
    ]

    extracted_parameters: list[dict[str, object]] = []

    number_pattern = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"

    for definition in parameter_definitions:
        label_pattern = definition["label_pattern"]
        unit_pattern = definition["unit_pattern"]

        mention_match = re.search(
            label_pattern,
            text,
            flags=re.IGNORECASE,
        )

        if mention_match is None:
            continue

        value_pattern = (
            rf"{label_pattern}"
            rf".{{0,100}}?"
            rf"(?P<value>{number_pattern})"
            rf"\s*(?P<unit>{unit_pattern})"
        )

        value_match = re.search(
            value_pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        snippet_start = max(0, mention_match.start() - 60)
        snippet_end = min(len(text), mention_match.end() + 120)

        evidence = " ".join(
            text[snippet_start:snippet_end].split()
        )

        if value_match:
            raw_value = value_match.group("value").replace(",", ".")
            unit = value_match.group("unit")

            try:
                value: float | None = float(raw_value)
            except ValueError:
                value = None

            status = "numeric_value_extracted"
        else:
            value = None
            unit = definition["default_unit"]
            status = "mentioned_without_numeric_value"

        extracted_parameters.append(
            {
                "name": definition["name"],
                "value": value,
                "unit": unit,
                "status": status,
                "evidence": evidence,
            }
        )

    formulas = re.findall(
        r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[^,.;\n]+",
        text,
    )

    return {
        "status": "success",
        "parameters_found": len(extracted_parameters),
        "parameters": extracted_parameters,
        "formulas": formulas,
        "note": (
            "Parameters without explicit numerical values are stored "
            "with value null instead of being invented."
        ),
    }

root_agent = Agent(
    model="gemini-3.5-flash",

    # Automatische Wiederholungsversuche bei vorübergehenden
    # Netzwerk-, Kapazitäts- und Serverfehlern
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                initial_delay=2.0,
                attempts=8,
                exp_base=2.0,
                max_delay=30.0,
                http_status_codes=[
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                ],
            ),
            timeout=120_000,
        )
    ),

    name="scholar_prime",

    description=(
        "An academic research agent specialized in querying "
        "scientific databases and extracting material parameters."
    ),

    instruction=(
        "You are Scholar-Prime, a professional academic research "
        "agent. Use the search_arxiv tool whenever the user requests "
        "scientific literature from arXiv. Evaluate the relevance of "
        "the returned abstracts, identify the most relevant paper, "
        "and summarize its scientific contribution accurately. "
        "Extract important formulas and material parameters when "
        "they are present. Always state the DOI of every cited paper "
        "when a DOI is available. Never invent a DOI, formula, "
        "parameter, author, or paper result. Clearly distinguish "
        "information taken directly from a paper from your own "
        "interpretation. Call search_arxiv only once per user request "
        "unless the tool returns an error or no papers. Request at "
        "most three results and do not repeat the search merely to "
        "refine the wording."
    ),

    tools=[
    search_arxiv,
    extract_parameters_from_text,
    ],
)