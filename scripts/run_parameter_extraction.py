from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# Hauptverzeichnis des Projekts bestimmen
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sicherstellen, dass scholar_prime importiert werden kann,
# auch wenn dieses Skript aus dem scripts-Ordner gestartet wird.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scholar_prime.agent import (  # noqa: E402
    extract_parameters_from_text,
    root_agent,
    search_arxiv,
)


SEARCH_QUERY = (
    "thermodynamic simulation parameters for advanced fission reactors"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "simulation_parameters.json"
)


def parse_json_objects(raw_output: str) -> list[dict[str, Any]]:
    """
    Extracts all valid JSON objects from the arXiv CLI output.

    The Science-Skills arXiv script may print several consecutive JSON
    objects while collecting results. This function parses every valid
    object so that the final and most complete result can be selected.

    Args:
        raw_output: Complete console output returned by search_arxiv.

    Returns:
        A list containing all successfully parsed JSON objects.

    Raises:
        ValueError: If no valid JSON object can be found.
    """

    decoder = json.JSONDecoder()
    parsed_objects: list[dict[str, Any]] = []
    position = 0

    while position < len(raw_output):
        object_start = raw_output.find("{", position)

        if object_start == -1:
            break

        try:
            parsed_object, object_end = decoder.raw_decode(
                raw_output,
                object_start,
            )
        except json.JSONDecodeError:
            position = object_start + 1
            continue

        if isinstance(parsed_object, dict):
            parsed_objects.append(parsed_object)

        position = object_end

    if not parsed_objects:
        raise ValueError(
            "The arXiv search output did not contain valid JSON."
        )

    return parsed_objects


def select_complete_search_result(
    search_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Selects the successful search result containing the most papers.

    Args:
        search_results: Parsed JSON objects returned by the arXiv CLI.

    Returns:
        The successful result with the largest papers list.

    Raises:
        ValueError: If no successful result containing papers exists.
    """

    successful_results = [
        result
        for result in search_results
        if result.get("status") == "success"
        and isinstance(result.get("papers"), list)
        and result["papers"]
    ]

    if not successful_results:
        raise ValueError(
            "The arXiv search did not return any papers."
        )

    return max(
        successful_results,
        key=lambda result: len(result["papers"]),
    )


def calculate_relevance_score(
    paper: dict[str, Any],
) -> int:
    """
    Calculates a simple relevance score for a scientific paper.

    Papers containing terms related to reactor simulation, fission,
    thermodynamics, material properties, and simulation parameters
    receive a higher score.

    Args:
        paper: Paper metadata containing a title and abstract.

    Returns:
        Integer relevance score.
    """

    title = str(paper.get("title", ""))
    abstract = str(paper.get("summary", ""))

    searchable_text = f"{title} {abstract}".lower()

    keyword_weights = {
        "reactor": 6,
        "thermodynamic": 6,
        "thermal": 5,
        "simulation": 5,
        "parameter": 4,
        "fission": 4,
        "energy density": 4,
        "barrier height": 4,
        "material": 3,
        "conductivity": 3,
        "temperature": 3,
        "dynamics": 2,
        "actinide": 2,
    }

    score = sum(
        weight
        for keyword, weight in keyword_weights.items()
        if keyword in searchable_text
    )

    # Ein Paper mit DOI ist für die spätere Verifikation besser geeignet.
    if paper.get("doi"):
        score += 2

    return score


def select_most_relevant_paper(
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Selects the paper with the highest calculated relevance score.

    Args:
        papers: Papers returned by the arXiv search.

    Returns:
        The paper considered most relevant for the search objective.

    Raises:
        ValueError: If the paper list is empty.
    """

    if not papers:
        raise ValueError("No papers are available for selection.")

    return max(
        papers,
        key=calculate_relevance_score,
    )


def main() -> None:
    """
    Runs the complete literature search and parameter extraction chain.
    """

    print("=== SCHOLAR-PRIME PARAMETER EXTRACTION ===")
    print(f"Agent: {root_agent.name}")
    print(f"Query: {SEARCH_QUERY}")

    print("\nSearching arXiv...")

    raw_search_output = search_arxiv(
        query=SEARCH_QUERY,
        max_results=3,
    )

    if raw_search_output.startswith("Error:"):
        raise RuntimeError(raw_search_output)

    parsed_results = parse_json_objects(raw_search_output)

    complete_result = select_complete_search_result(
        parsed_results
    )

    papers = complete_result["papers"]

    selected_paper = select_most_relevant_paper(papers)

    title = selected_paper.get("title")
    abstract = selected_paper.get("summary")
    doi = selected_paper.get("doi")

    if not abstract:
        raise ValueError(
            "The selected paper does not contain an abstract."
        )

    print("\nMost relevant paper:")
    print(f"Title: {title}")
    print(f"DOI: {doi or 'No DOI available'}")
    print(
        "Relevance score:",
        calculate_relevance_score(selected_paper),
    )

    print("\nExtracting simulation parameters...")

    extraction_result = extract_parameters_from_text(
        abstract
    )

    output_data = {
        "agent": root_agent.name,
        "search_query": SEARCH_QUERY,
        "source_paper": {
            "title": title,
            "authors": selected_paper.get("authors", []),
            "doi": doi,
            "arxiv_id": selected_paper.get("id"),
            "pdf_url": selected_paper.get("pdf_url"),
            "published": selected_paper.get("published"),
            "abstract": abstract,
        },
        "extraction": extraction_result,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            output_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nExtraction completed successfully.")
    print(f"Output file: {OUTPUT_FILE}")

    print("\n=== GENERATED JSON ===")
    print(
        json.dumps(
            output_data,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()