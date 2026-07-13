"""OpenAlex research with an explicit, reproducible local fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from collective_lab_demo.models.parameter_models import SourceRecord


OPENALEX_ENDPOINT = "https://api.openalex.org/works"
SEARCH_TITLE = "Thermal conductivities of some novel nonlinear optical materials"


@dataclass(frozen=True, slots=True)
class LiteratureResult:
    """Source plus transparent information about the actual retrieval mode."""

    source: SourceRecord
    requested_mode: str
    actual_mode: str
    fallback_reason: str | None = None


class LiteratureService:
    """Retrieve a real OpenAlex record or load the checked-in fixture."""

    def __init__(self, fixture_path: str | Path | None = None) -> None:
        default = Path(__file__).resolve().parent.parent / "fixtures" / (
            "thermal_conductivity_source.json"
        )
        self.fixture_path = Path(fixture_path) if fixture_path else default

    def research(self, mode: str = "demo") -> LiteratureResult:
        """Load a fixture in demo mode; use OpenAlex first in online mode."""

        if mode == "demo":
            return LiteratureResult(
                source=self._load_fixture(),
                requested_mode=mode,
                actual_mode="fixture",
            )
        if mode != "online":
            raise ValueError("Mode must be either 'demo' or 'online'.")

        try:
            source = self._research_openalex()
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            return LiteratureResult(
                source=self._load_fixture(),
                requested_mode=mode,
                actual_mode="fixture-fallback",
                fallback_reason=str(error),
            )
        return LiteratureResult(
            source=source,
            requested_mode=mode,
            actual_mode="online-openalex",
        )

    def _load_fixture(self) -> SourceRecord:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return SourceRecord.model_validate(data)

    def _research_openalex(self) -> SourceRecord:
        query = urlencode(
            {
                "search": SEARCH_TITLE,
                "per-page": 5,
                "select": (
                    "id,doi,title,publication_year,authorships,"
                    "abstract_inverted_index"
                ),
            }
        )
        request = Request(
            f"{OPENALEX_ENDPOINT}?{query}",
            headers={
                "Accept": "application/json",
                "User-Agent": "genesis-oracle-collective-lab/1.0",
            },
        )
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("OpenAlex returned no works.")

        matching = next(
            (
                work
                for work in results
                if str(work.get("title", "")).casefold() == SEARCH_TITLE.casefold()
            ),
            None,
        )
        if not isinstance(matching, dict):
            raise ValueError("OpenAlex did not return the expected source work.")

        abstract = self._rebuild_abstract(matching.get("abstract_inverted_index"))
        value_match = re.search(
            r"0[.]015\s*W\s*/\s*\(?cm\s*K\)?",
            abstract,
            flags=re.IGNORECASE,
        )
        if value_match is None:
            raise ValueError(
                "The OpenAlex abstract contains no verified 0.015 W/(cm*K) value."
            )
        excerpt = self._excerpt_around(abstract, value_match.start(), value_match.end())

        authorships = matching.get("authorships") or []
        authors = [
            str(item.get("author", {}).get("display_name"))
            for item in authorships
            if item.get("author", {}).get("display_name")
        ]
        doi = matching.get("doi")
        url = str(doi or matching.get("id") or "")
        if not url:
            raise ValueError("The OpenAlex work has no traceable URL.")

        return SourceRecord(
            title=str(matching["title"]),
            authors=authors,
            publication_year=int(matching["publication_year"]),
            url=url,
            doi=str(doi).removeprefix("https://doi.org/") if doi else None,
            parameter_name="thermal_conductivity_AgGaS2_perpendicular",
            original_value=0.015,
            original_unit="W/(cm*K)",
            excerpt=excerpt,
            provider="OpenAlex",
            retrieval_mode="online",
            is_fixture=False,
        )

    @staticmethod
    def _rebuild_abstract(index: Any) -> str:
        if not isinstance(index, dict) or not index:
            raise ValueError("The OpenAlex work has no abstract for extraction.")
        positioned_words: list[tuple[int, str]] = []
        for word, positions in index.items():
            if isinstance(positions, list):
                positioned_words.extend((int(position), str(word)) for position in positions)
        return " ".join(word for _, word in sorted(positioned_words))

    @staticmethod
    def _excerpt_around(text: str, start: int, end: int) -> str:
        left = max(0, start - 180)
        right = min(len(text), end + 180)
        return " ".join(text[left:right].split())
