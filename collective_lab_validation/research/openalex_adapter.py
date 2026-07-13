"""Direct OpenAlex API fallback and metadata-to-source conversion."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from collective_lab_validation.models.parameter_payload import SourceRecord


SOURCE_DOI = "10.1364/AO.33.001000"
SOURCE_DOI_URL = f"https://doi.org/{SOURCE_DOI}"
SOURCE_TITLE = "Thermal conductivities of some novel nonlinear optical materials"
SELECT_FIELDS = (
    "id,doi,display_name,publication_year,authorships,abstract_inverted_index"
)


def rebuild_abstract(index: Any) -> str:
    if not isinstance(index, dict) or not index:
        raise ValueError("OpenAlex record has no abstract evidence.")
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        if isinstance(positions, list):
            positioned.extend((int(position), str(word)) for position in positions)
    return " ".join(word for _, word in sorted(positioned))


def source_from_openalex(
    data: dict[str, Any],
    *,
    retrieval_mode: str,
    science_skill_used: str | None,
) -> SourceRecord:
    abstract = rebuild_abstract(data.get("abstract_inverted_index"))
    parallel = re.search(r"0[.]014\s*W\s*/\s*\(?cm\s*K\)?", abstract, re.I)
    perpendicular = re.search(r"0[.]015\s*W\s*/\s*\(?cm\s*K\)?", abstract, re.I)
    if parallel is None or perpendicular is None:
        raise ValueError("OpenAlex abstract does not contain both AgGaS2 values.")
    left = max(0, parallel.start() - 180)
    right = min(len(abstract), perpendicular.end() + 220)
    excerpt = " ".join(abstract[left:right].split())
    authors = [
        str(item.get("author", {}).get("display_name"))
        for item in data.get("authorships", [])
        if item.get("author", {}).get("display_name")
    ]
    doi_url = str(data.get("doi") or SOURCE_DOI_URL)
    return SourceRecord(
        title=str(data.get("display_name") or SOURCE_TITLE),
        authors=authors,
        publication_year=int(data.get("publication_year") or 1994),
        doi=doi_url.removeprefix("https://doi.org/"),
        source_url=doi_url,
        source_repository="OpenAlex",
        source_excerpt=excerpt,
        parameter_names=[
            "AgGaS2 thermal conductivity parallel to optic axis",
            "AgGaS2 thermal conductivity perpendicular to optic axis",
        ],
        source_values={"parallel": 0.014, "perpendicular": 0.015},
        source_unit="W/(cm*K)",
        retrieval_mode=retrieval_mode,
        science_skill_used=science_skill_used,
    )


class OpenAlexAdapter:
    """Technical fallback; never represented as a DeepMind Science Skill."""

    def fetch_known_source(self) -> SourceRecord:
        identifier = quote(SOURCE_DOI_URL, safe="")
        query = urlencode({"select": SELECT_FIELDS})
        request = Request(
            f"https://api.openalex.org/works/{identifier}?{query}",
            headers={
                "Accept": "application/json",
                "User-Agent": "genesis-oracle-collective-validation/1.0",
            },
        )
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return source_from_openalex(
            data,
            retrieval_mode="online-openalex-api-fallback",
            science_skill_used=None,
        )
