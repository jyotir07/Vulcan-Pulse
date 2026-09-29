"""Live smoke run of the natural-language parser against the configured LLM.

Sends every phrasing in tests/data/nl_phrasings.json to the real provider and compares the
parsed scenario with the expected one. Needs VULCAN_LLM_API_KEY (and optionally
VULCAN_LLM_MODEL); costs a few cents.

Usage:
    python scripts/nl_smoke.py
"""

import json
import sys

from backend.config import REPO_ROOT
from backend.services.nl_parser import parser_from_env
from backend.simulation.scenarios import Scenario

PHRASINGS = REPO_ROOT / "tests" / "data" / "nl_phrasings.json"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = parser_from_env()
    if parser.client is None:
        raise SystemExit("Set VULCAN_LLM_API_KEY to run the live smoke test.")
    cases = json.loads(PHRASINGS.read_text(encoding="utf-8"))
    passed = 0
    for case in cases:
        result = parser.parse(case["text"])
        if case["expected"] == "clarify":
            ok = result.status == "clarify"
            got = result.message
        else:
            expected = Scenario.model_validate({"interventions": case["expected"]})
            ok = result.status == "parsed" and result.scenario == expected
            got = (
                result.scenario.model_dump(mode="json", by_alias=True, exclude_defaults=True)
                if result.scenario
                else result.message
            )
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['text']}")
        if not ok:
            print(f"      got: {got}")
    print(f"\n{passed}/{len(cases)} phrasings parsed as expected")


if __name__ == "__main__":
    main()
