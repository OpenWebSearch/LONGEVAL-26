#!/usr/bin/env python3
from __future__ import annotations

import gzip
from glob import glob
from pathlib import Path

from tqdm import tqdm
import click
from ranx import Run, fuse
from trectools import TrecQrel


def resolve_qrel_paths(glob_pattern: str) -> list[Path]:
    return [Path(path) for path in sorted(glob(glob_pattern, recursive=True))]


def qrels_to_run(path: Path, min_rel: int) -> Run:
    qrels = TrecQrel(str(path)).qrels_data.copy()
    qrels["query"] = qrels["query"].astype(str)
    qrels["docid"] = qrels["docid"].astype(str)
    qrels = qrels[qrels["rel"] >= min_rel]

    if qrels.empty:
        raise click.ClickException(
            f"{path} has no documents with relevance >= {min_rel}."
        )

    qrels = qrels.sort_values(
        by=["query", "rel", "docid"], ascending=[True, False, True]
    )

    run = {
        query_id: {
            row.docid: float(row.rel)
            for row in query_rows.itertuples(index=False)
        }
        for query_id, query_rows in qrels.groupby("query", sort=False)
    }

    return Run(run=run, name=path.stem)


def align_queries(runs: list[Run]) -> list[Run]:
    all_query_ids = sorted({query_id for run in runs for query_id in run.keys()})

    return [
        Run(
            run={
                query_id: dict(run_dict.get(query_id, {}))
                for query_id in all_query_ids
            },
            name=run.name,
        )
        for run in runs
        for run_dict in [run.to_dict()]
    ]


def save_trec_run(run: Run, output: Path) -> None:
    if output.suffix != ".gz":
        run.save(str(output), kind="trec")
        return

    if not run.sorted:
        run.sort()

    run_dict = run.to_dict()
    lines = [
        f"{query_id} Q0 {doc_id} {rank} {score} {run.name}"
        for query_id, docs in run_dict.items()
        for rank, (doc_id, score) in enumerate(docs.items(), start=1)
    ]

    with gzip.open(output, "wt", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))


@click.command()
@click.option(
    "--glob",
    "glob_pattern",
    required=True,
    type=str,
    help='Glob that selects the qrel files, e.g. "llm-qrels/*snapshot-1*".',
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the fused run in TREC format. Use a .gz suffix for gzip output.",
)
@click.option(
    "--min-rel",
    default=1,
    show_default=True,
    type=int,
    help="Only documents with rel >= min-rel are kept when converting qrels to runs.",
)
@click.option(
    "--rrf-k",
    default=60,
    show_default=True,
    type=int,
    help="RRF k parameter passed to ranx.",
)
def main(glob_pattern: str, output: Path, min_rel: int, rrf_k: int) -> None:
    qrel_paths = resolve_qrel_paths(glob_pattern)

    if not qrel_paths:
        raise click.ClickException(f'No qrel files matched glob "{glob_pattern}".')

    runs = [qrels_to_run(path, min_rel=min_rel) for path in tqdm(qrel_paths, "load runs")]
    runs = align_queries(runs)

    fused_run = fuse(runs=runs, method="rrf", params={"k": rrf_k})
    fused_run.name = "rrf-fused-qrels"

    output.parent.mkdir(parents=True, exist_ok=True)
    save_trec_run(fused_run, output)

    click.echo(
        f"Fused {len(runs)} qrel files with RRF and wrote the run to {output}."
    )


if __name__ == "__main__":
    main()
