"""Validate a single revision-bound frozen contract for worker dispatch."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from run import QualityGateError, validate_frozen_contract_artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--revision", required=True)
    return parser.parse_args(argv)


def _read_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise QualityGateError(f"unable to read contract: {error}") from error
    except json.JSONDecodeError as error:
        raise QualityGateError(f"invalid contract JSON: {error}") from error
    if not isinstance(payload, dict):
        raise QualityGateError("contract root must be an object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.require_frozen:
        print("contract validation error: --require-frozen is required", file=sys.stderr)
        return 2

    try:
        validate_frozen_contract_artifact(_read_contract(args.contract), expected_revision=args.revision)
    except QualityGateError as error:
        print(f"contract validation error: {error}", file=sys.stderr)
        return 2

    print(f"{args.contract}: frozen contract validated at revision {args.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
