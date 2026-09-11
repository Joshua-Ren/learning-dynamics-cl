#!/usr/bin/env python
from pathlib import Path
import json
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.sequence_score.validation import run_closed_form_validations


def main() -> None:
    results = run_closed_form_validations()
    print(json.dumps(results, indent=2, sort_keys=True))
    if not all(item["ok"] for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
