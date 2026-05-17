#!/usr/bin/env bash

set -e

for SNAP in "snapshot-1" "snapshot-2" "snapshot-3"
do
    echo $SNAP
    ./fuse_qrels.py --glob "llm-qrels/*$SNAP.qrels.txt" --output rrf-relevance-feedback-$SNAP.run.txt.gz
done