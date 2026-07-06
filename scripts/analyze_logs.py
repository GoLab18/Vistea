"""Results export: tables from the JSONL research log.

Usage:
    uv run python scripts/analyze_logs.py [logs/vistea.jsonl]

Reads events from the log and saves three CSVs alongside it:
  *_sessions.csv    - one line per session: strategy, grid, mean error
                      (validation_report + context bound with bind()),
  *_points.csv      - error per validation point (validation_point) - data
                      for heatmaps and upper-row bias verification,
  *_strategies.csv  - strategy x grid_size aggregate: mean/std/min/max/n -
                      directly serves as the length <-> accuracy trade-off table.

The console prints the strategy aggregate (preview without opening CSVs).
"""

from __future__ import annotations

import sys

import pandas as pd


def main(path: str = "logs/vistea.jsonl") -> None:
    df = pd.read_json(path, lines=True)
    base = path.rsplit(".", 1)[0]

    # per session: validation reports with session context
    rep = df[df["event"] == "validation_report"].copy()
    if rep.empty:
        print("No validation_report events found in", path)
        return
    session_cols = [
        c for c in (
            "timestamp", "session_id", "strategy", "grid_size",
            "lighting_condition", "preprocessing",
            "mean_error_px", "mean_error_deg", "mean_error_percent", "n_points",
        ) if c in rep.columns
    ]
    sessions = rep[session_cols].sort_values("timestamp")
    sessions.to_csv(f"{base}_sessions.csv", index=False)

    # per point: validation errors (heatmap / bias)
    pts = df[df["event"] == "validation_point"].copy()
    if not pts.empty:
        pts["px_x"] = pts["point"].str[0]
        pts["px_y"] = pts["point"].str[1]
        point_cols = [
            c for c in (
                "timestamp", "session_id", "strategy", "grid_size",
                "px_x", "px_y", "error_px", "error_deg",
            ) if c in pts.columns
        ]
        pts[point_cols].to_csv(f"{base}_points.csv", index=False)

        per_point = (
            pts.groupby(["px_x", "px_y"])["error_deg"]
            .agg(["mean", "std", "count"]).round(3)
        )
        print("\nError per validation point (deg):\n", per_point.to_string())

    # aggregate: strategy x grid (core of the "Tests" chapter)
    group_keys = [k for k in ("strategy", "grid_size") if k in rep.columns]
    agg = (
        rep.groupby(group_keys)["mean_error_deg"]
        .agg(["mean", "std", "min", "max", "count"]).round(3)
        .sort_values("mean")
    )
    agg.to_csv(f"{base}_strategies.csv")
    print("\nStrategy x grid (mean_error_deg):\n", agg.to_string())
    print(f"\nSaved: {base}_sessions.csv, {base}_points.csv, {base}_strategies.csv")


if __name__ == "__main__":
    main(*sys.argv[1:2])