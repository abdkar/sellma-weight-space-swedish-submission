"""Verify and reanalyze compact Colab outputs on CPU."""

import argparse
import csv
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from sellma_wsp.analysis import analyze
from sellma_wsp.common import append_jsonl, digest, file_hash, load_json, read_jsonl
from sellma_wsp.experiment import config_hash, phase_dir
from sellma_wsp.scoring import score, sections


RUNS = {"test": "test-370e9d84ae06", "pilot": "pilot-fe4f8fab8ef7"}
ANALYSIS_FILES = (
    "summary.csv", "paired_effects.csv", "budget_curves.csv", "coverage.json",
    "rescues.json", "swedish_success.png", "budget_curves.png",
)


def numeric_only(row):
    """Exploratory arithmetic score used in the final PDF's numeric-only column."""
    if not row["endpoint"]["completed"]:
        return False
    output = row["output"]
    if row["thinking"]:
        _, marker, output = output.rpartition("</think>")
        if not marker:
            return False
    integers = re.findall(r"\d+", output.replace(",", ""))
    return bool(integers) and integers[-1] == str(row["gold"])


def write_numeric_summary(path, rows):
    outcomes = defaultdict(list)
    for row in rows:
        if row["split"] in {"test", "dev"}:
            outcomes[row["split"], row["language"], row["condition"]].append(numeric_only(row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["split", "language", "condition", "n", "numeric_correct_posthoc"])
        for group, successes in sorted(outcomes.items()):
            writer.writerow([*group, len(successes), sum(successes)])


def restore_rows(rows, config):
    expected_hash = config_hash(config)
    for row in rows:
        if row["config_hash"] != expected_hash or row["dataset_hash"] != config["dataset_hash"]:
            raise ValueError(f"Prediction fingerprint mismatch for {row['id']}")
        args = (row["output"], row["thinking"], row["gold"], row["answer_type"],
                row["finish_reason"], row["tokens"], config["cap"])
        if score(*args) != row["endpoint"]:
            raise ValueError(f"Recorded score differs for {row['id']} in condition {row['condition']}")
        if "reasoning_prefix250" in row:
            reasoning, final, _ = sections(row["output"], row["thinking"])
            row["language_views"] = {
                "prefix250": row.pop("reasoning_prefix250"),
                "reasoning": reasoning,
                "final": final,
            }
        else:
            row["language_views"] = {}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=RUNS, default="test")
    parser.add_argument("--output", type=Path, default=Path("reanalysis"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / "results" / RUNS[args.phase]
    output = args.output.resolve() / source.name
    if output.exists():
        parser.error(f"Output already exists: {output}")
    config = load_json(source / "config.json")
    manifest = {
        path.relative_to(root).as_posix(): file_hash(path)
        for directory in ("sellma_wsp", "vendor/upstream")
        for path in (root / directory).glob("*.py")
    }
    if digest(manifest) != config["code_hash"]:
        raise ValueError("Source fingerprint differs from the recorded run")
    if digest(read_jsonl(root / "data/tasks.jsonl")) != config["dataset_hash"]:
        raise ValueError("Task fingerprint differs from the recorded run")

    rows = restore_rows(read_jsonl(source / "predictions_compact.jsonl"), config)
    with tempfile.TemporaryDirectory(prefix="sellma-reanalysis-") as staging:
        config.update(root=str(root), run_dir=staging, backup_dir=None)
        target = phase_dir(config)
        if target.name != source.name:
            raise ValueError("Configuration fingerprint differs from the recorded run")
        append_jsonl(target / "predictions.jsonl", rows)
        _, _, coverage = analyze(config)
        if not coverage["complete"]:
            raise ValueError(f"Incomplete coverage: {coverage}")
        output.mkdir(parents=True)
        for name in ANALYSIS_FILES:
            shutil.copyfile(target / name, output / name)
        write_numeric_summary(output / "numeric_only_posthoc.csv", rows)
    print(f"Verified source and tasks. Reanalyzed {coverage['completed']} saved predictions in {output}")


if __name__ == "__main__":
    main()
