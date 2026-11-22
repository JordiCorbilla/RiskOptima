"""Enforce independent statement and branch coverage floors from coverage JSON."""

import json
import sys
from pathlib import Path

FLOORS = {"lines": (65.52, "covered_lines", "num_statements"), "branches": (53.70, "covered_branches", "num_branches")}


def main(report: Path) -> int:
    coverage = json.loads(report.read_text(encoding="utf-8"))
    totals = coverage["totals"]
    if not coverage["meta"]["branch_coverage"] or not totals["num_branches"]:
        raise ValueError("a coverage report with branch measurement is required")
    passed = True
    for name, (floor, covered, total) in FLOORS.items():
        actual = 100.0 * totals[covered] / totals[total]
        print(f"{name}: {actual:.4f}% ({totals[covered]}/{totals[total]}), minimum {floor:.2f}%")
        passed &= actual >= floor
    return int(not passed)


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
