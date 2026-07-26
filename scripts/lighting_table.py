"""Accuracy vs (lighting x preprocessing).

Usage:
    uv run python scripts/lighting_table.py [logs/vistea.jsonl]

Reads validation_report events from the JSONL log and prints a pivot:
rows = lighting_condition, columns = preprocessing, values = mean
error in degrees (n sessions in parentheses). The same CSV file lands next to the log.
"""

from __future__ import annotations

import sys
import pandas as pd


def main(path: str = "logs/vistea.jsonl") -> None:
    df = pd.read_json(path, lines=True)
    rep = df[df["event"] == "validation_report"].copy()
    if rep.empty:
        print("No validation_report events in", path)
        return

    for col, default in (("lighting_condition", "unknown"), ("preprocessing", "none")):
        if col not in rep:
            rep[col] = default
        rep[col] = rep[col].fillna(default)

    table = rep.pivot_table(
        index="lighting_condition",
        columns="preprocessing",
        values="mean_error_deg",
        aggfunc=["mean", "count"],
    )
    print(table.round(2).to_string())

    out = path.rsplit(".", 1)[0] + "_lighting_table.csv"
    table.to_csv(out)
    print("\nSaved:", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])