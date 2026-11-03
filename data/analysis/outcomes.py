"""Summarize existing source outcomes without exporting session-level labels.

This module does not grade conversations or alter annotation packets. It reads
only source session/score tables and curated metadata, never conversation text.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

REPORT = Path(__file__).parent / "reports/outcome_summary.json"
QUALITY_DIMENSIONS = [
    ("Correctness & Accuracy", 40, "Accuracy of facts, calculations, logic and source use"),
    ("Deliverable Completeness", 20, "Presence of required files, sections and components"),
    ("Task Requirements", 25, "Compliance with task-specific instructions and constraints"),
    ("Technical Quality", 15, "Formatting, organization, usability and technical execution"),
]
SUCCESS_DIMENSIONS = [
    ("Goal completion", "Resolution of each distinct user request"),
    ("Final session state", "Clean conclusion versus unresolved errors or abandonment"),
    ("Agent efficiency", "Steady progress versus repeated errors and wasted actions"),
    ("Code and commit quality", "Evidence of correct, complete code and sound commits"),
    ("User experience", "Satisfaction, friction and correction effort"),
]
# Bands come from SWE-chat Appendix F.2.5. Half-open intervals also retain
# fractional values that occur in the released table despite its integer rubric.
SUCCESS_BANDS = [
    (0, 10, "0-9", "No useful progress"),
    (10, 30, "10-29", "Very little useful work"),
    (30, 50, "30-49", "Relevant output; primary goal unmet"),
    (50, 70, "50-69", "Meaningful progress; significant work or defects remain"),
    (70, 90, "70-89", "Core goal met; minor issues remain"),
    (90, 101, "90-100", "All requests resolved; clean ending"),
]


def success_band(value):
    if value is None or pd.isna(value):
        return "Missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Invalid"
    if not math.isfinite(number) or not 0 <= number <= 100:
        return "Invalid"
    return next(label for low, high, label, _ in SUCCESS_BANDS if low <= number < high)


def summarize_scores(values):
    raw = pd.Series(list(values), dtype="object")
    missing = raw.isna() | raw.astype("string").str.strip().eq("").fillna(False)
    numeric = pd.to_numeric(raw, errors="coerce")
    valid = numeric.notna() & numeric.between(0, 100)
    scored = numeric[valid].astype(float)
    stats = {"sessions": len(raw), "scored": int(valid.sum()),
             "missing": int(missing.sum()), "invalid": int((~missing & ~valid).sum()),
             "fractional_scores": int(scored.mod(1).ne(0).sum())}
    for name, value in {"mean": scored.mean(), "median": scored.median(),
                        "minimum": scored.min(), "maximum": scored.max(),
                        "p25": scored.quantile(.25), "p75": scored.quantile(.75)}.items():
        stats[name] = None if pd.isna(value) else round(float(value), 3)
    bins = pd.cut(scored, bins=[*range(0, 100, 10), 101], right=False)
    stats["histogram_10_point_counts"] = [int(v) for v in bins.value_counts(sort=False)]
    stats["histogram_intervals"] = [f"[{i}, {i+10})" for i in range(0, 90, 10)] + ["[90, 100]"]
    counts = Counter(success_band(v) for v in raw.mask(missing, None))
    stats["swe_rubric_bands"] = {label: counts[label] for _, _, label, _ in SUCCESS_BANDS}
    return stats


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(source_root, curated_root):
    source_root, curated_root = Path(source_root), Path(curated_root)
    native = source_root / "raw/collabskill/leaderboard_data"
    sessions = json.loads((native / "sessions.json").read_text())
    details = json.loads((native / "scores.json").read_text())
    if len({r["session_id"] for r in sessions}) != len(sessions):
        raise ValueError("Duplicate source CollabSkill session IDs.")
    detail_index = {row["_id"]: row for row in details}
    swe_path = source_root / "raw/swe-chat/sessions.parquet"
    swe = pd.read_parquet(swe_path, columns=["session_id", "session_success", "summary_outcome"])
    if swe.session_id.duplicated().any():
        raise ValueError("Duplicate source SWE-chat session IDs.")
    manifest_path = source_root / "curation/config/download_manifest.json"
    if not manifest_path.exists():
        manifest_path = source_root / "download_manifest.json"  # Original local workspace layout.
    manifest = json.loads(manifest_path.read_text())
    inputs = [native / "sessions.json", native / "scores.json", swe_path, manifest_path]
    report = {"schema_version": "1.0", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Aggregate source and saved-sample outcomes; no session IDs, source text or human judgments",
              "source_revisions": {name: row["revision"] for name, row in manifest["datasets"].items()},
              "quality_dimensions": [dict(dimension=n, usual_maximum=m, meaning=d) for n, m, d in QUALITY_DIMENSIONS],
              "success_dimensions": [dict(dimension=n, meaning=d) for n, d in SUCCESS_DIMENSIONS],
              "success_bands": [dict(band=b, meaning=d) for _, _, b, d in SUCCESS_BANDS],
              "sources": {"CollabSkill": "https://arxiv.org/html/2606.09833v2#S3.SS2",
                          "SWE-chat": "https://arxiv.org/html/2604.20779v2#A6.SS2.SSS5"},
              "populations": {}, "input_sha256": {}}
    for label, folder, source in [("CollabSkill", "collabskill-curated", sessions),
                                   ("SWE-chat", "swe_chat_curated", swe)]:
        sample_path = curated_root / folder / "scenario_manifest.csv"
        sample = pd.read_csv(sample_path)
        if sample.session_id.duplicated().any():
            raise ValueError("Duplicate curated session IDs.")
        available = {r["session_id"] for r in source} if label == "CollabSkill" else set(source.session_id)
        if not set(sample.session_id) <= available:
            raise ValueError("Curated IDs missing from source sessions.")
        report["input_sha256"]["curated/" + folder + "/scenario_manifest.csv"] = sha256(sample_path)
        populations = {"source": available, "curated": set(sample.session_id)}
        populations.update({"curated_"+str(turn): set(g.session_id) for turn, g in sample.groupby("turn_type")})
        report["populations"][label] = {}
        for population, ids in populations.items():
            if label == "CollabSkill":
                retained = [r for r in sessions if r["session_id"] in ids]
                stats = summarize_scores(r.get("autograde_overall_score") for r in retained)
                stats.pop("swe_rubric_bands")  # Never apply SWE's semantic bands to CollabSkill.
                records = [detail_index[r["autograde_score_id"]] for r in retained
                           if r.get("autograde_score_id") in detail_index]
                stats["detailed_score_records"] = len(records)
                stats["standard_four_records"] = sum(
                    {v["criterion_name"] for v in r.get("detailed_scores", [])} == {n for n, _, _ in QUALITY_DIMENSIONS}
                    for r in records)
                stats["criterion_summary"] = []
                for name, maximum, _ in QUALITY_DIMENSIONS:
                    values = [v["score"] for r in records for v in r.get("detailed_scores", []) if v["criterion_name"] == name]
                    stats["criterion_summary"].append(dict(dimension=name, usual_maximum=maximum,
                        records=len(values), mean_points=None if not values else round(sum(values)/len(values), 3)))
            else:
                retained = swe[swe.session_id.isin(ids)]
                stats = summarize_scores(retained.session_success)
                stats["nonempty_outcome_summaries"] = int(retained.summary_outcome.fillna("").str.strip().ne("").sum())
            report["populations"][label][population] = stats
    report["input_sha256"].update({str(path.relative_to(source_root)): sha256(path) for path in inputs})
    return report


def load_report(path=REPORT):
    return json.loads(Path(path).read_text())


def coverage_table(report):
    return pd.DataFrame([dict(dataset=name, population=population,
        **{k: values.get(k) for k in ["sessions", "scored", "missing", "invalid", "mean", "median", "detailed_score_records", "nonempty_outcome_summaries"]})
        for name, populations in report["populations"].items()
        for population, values in populations.items() if population in {"source", "curated"}])


def plot_outcomes(report):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, name in zip(axes, ["CollabSkill", "SWE-chat"]):
        for population, color in [("source", "#326b91"), ("curated", "#c27a32")]:
            values = report["populations"][name][population]
            counts = values["histogram_10_point_counts"]
            percentages = [100*n/values["scored"] if values["scored"] else 0 for n in counts]
            ax.stairs(percentages, [*range(0, 100, 10), 100], label=population.title(), color=color, linewidth=2)
        ax.set(title=name, xlabel="Source outcome score (0-100)", ylabel="% of scored sessions")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--curated-root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New aggregate JSON destination; existing files are preserved")
    args = parser.parse_args()
    report = build_report(args.source_root, args.curated_root or args.source_root / "curated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Wrote aggregate outcome report. Source tables and annotation files were not changed.")


if __name__ == "__main__":
    main()
