"""Validate Wave 5 evidence with a bounded human review and emit a recommendation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mochi.agents.adaptive_release_qualification import (
    evaluate_canary,
    load_canary_review,
    load_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence_bytes = args.evidence.read_bytes()
    review_bytes = args.review.read_bytes()
    decision = evaluate_canary(
        load_evidence(args.evidence),
        load_canary_review(args.review),
        evidence_document_bytes=evidence_bytes,
        review_document_bytes=review_bytes,
    )
    if args.output.resolve(strict=False) in {
        args.evidence.resolve(strict=False),
        args.review.resolve(strict=False),
    }:
        parser.error("--output must not overwrite evidence or review")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
