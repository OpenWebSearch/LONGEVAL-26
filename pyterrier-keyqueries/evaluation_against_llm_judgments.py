#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import click
from ir_measures import nDCG
import pandas as pd
import pyterrier as pt


DEFAULT_SNAPSHOTS = ("snapshot-1", "snapshot-2", "snapshot-3")
DEFAULT_QRELS_DIRECTORY = Path(__file__).parent / "llm-qrels"


def resolve_qrels_path(qrels_directory: Path, llm: str, snapshot: str) -> Path:
    matches = sorted(qrels_directory.glob(f"{llm}-*-{snapshot}.qrels.txt"))

    if not matches:
        raise FileNotFoundError(
            f'No qrels found for llm "{llm}" and snapshot "{snapshot}" in {qrels_directory}.'
        )

    if len(matches) > 1:
        raise ValueError(
            f'Ambiguous qrels for llm "{llm}" and snapshot "{snapshot}": {matches}'
        )

    return matches[0]


def resolve_run_path(run_directory: Path, snapshot: str) -> Path:
    snapshot_directory = run_directory / snapshot
    candidates = [
        snapshot_directory / "run.txt.gz",
        snapshot_directory / "run.txt",
        snapshot_directory / "run.trec.gz",
        snapshot_directory / "run.trec",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No run file found in {snapshot_directory}.")


def load_qrels(path: Path, snapshot: str) -> pd.DataFrame:
    qrels = pt.io.read_qrels(str(path)).copy()
    qrels["qid"] = snapshot + ":" + qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels["label"] = pd.to_numeric(qrels["label"])
    return qrels


def load_run(path: Path, snapshot: str) -> pd.DataFrame:
    run = pt.io.read_results(str(path), format="trec").copy()
    run["qid"] = snapshot + ":" + run["qid"].astype(str)
    run["docno"] = run["docno"].astype(str)
    run["rank"] = pd.to_numeric(run["rank"]).astype(int)
    run["score"] = pd.to_numeric(run["score"])
    return run


def create_topics(qrels: pd.DataFrame) -> pd.DataFrame:
    qids = sorted(qrels["qid"].astype(str).unique())
    return pd.DataFrame({"qid": qids, "query": [""] * len(qids)})


def evaluate_run(run: pd.DataFrame, qrels: pd.DataFrame, name: str) -> tuple[float, float]:
    topics = create_topics(qrels)
    ret = pt.Experiment(
        [run],
        topics,
        qrels,
        [nDCG@10, nDCG(judged_only=True)@10],
        names=[name],
        filter_by_topics=False,
    )


    assert len(ret) == 1

    return float(ret.loc[0, "nDCG@10"]), float(ret.loc[0, "nDCG(judged_only=True)@10"])


def evaluate_against_llm_judgments(
    run_directory: str | Path,
    llm: str,
    qrels_directory: str | Path = DEFAULT_QRELS_DIRECTORY,
    snapshots: Sequence[str] = DEFAULT_SNAPSHOTS,
) -> pd.DataFrame:
    run_directory = Path(run_directory)
    qrels_directory = Path(qrels_directory)

    rows = []
    combined_runs = []
    combined_qrels = []

    for snapshot in snapshots:
        run = load_run(resolve_run_path(run_directory, snapshot), snapshot)
        qrels = load_qrels(resolve_qrels_path(qrels_directory, llm, snapshot), snapshot)
        ndcg_10, ndcg_10_unjudged_removed = evaluate_run(run, qrels, run_directory.name)

        rows.append(
            {
                "method": run_directory.name,
                "llm": llm,
                "snapshot": snapshot,
                "nDCG@10": ndcg_10,
                "nDCG@10_unjudged_removed": ndcg_10_unjudged_removed,
            }
        )
        combined_runs.append(run)
        combined_qrels.append(qrels)

    combined_run = pd.concat(combined_runs, ignore_index=True)
    combined_qrels_df = pd.concat(combined_qrels, ignore_index=True)
    ndcg_10, ndcg_10_unjudged_removed = evaluate_run(
        combined_run, combined_qrels_df, run_directory.name
    )
    rows.append(
        {
            "method": run_directory.name,
            "llm": llm,
            "snapshot": "all",
            "nDCG@10": ndcg_10,
            "nDCG@10_unjudged_removed": ndcg_10_unjudged_removed,
        }
    )

    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--run-directory",
    required=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory that contains snapshot-1 to snapshot-3 runs.",
)
@click.option("--llm", required=True, type=str, help="LLM prefix in llm-qrels.")
@click.option(
    "--qrels-directory",
    default=DEFAULT_QRELS_DIRECTORY,
    show_default=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory that contains the LLM-generated qrels.",
)
def main(run_directory: Path, llm: str, qrels_directory: Path) -> None:
    evaluation = evaluate_against_llm_judgments(run_directory, llm, qrels_directory)
    click.echo(evaluation.to_csv(index=False))


if __name__ == "__main__":
    main()
