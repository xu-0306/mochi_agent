"""Run explicit, tool-disabled external-model qualification fixtures.

This command deliberately refuses to contact any model until the caller passes
``--allow-external-model``.  The output is a redacted evidence document; it
never contains prompts, completions, URLs, model names, or credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mochi.agents.adaptive_release_qualification import (
    ExternalQualificationRunner,
    load_external_qualification_fixtures,
)
from mochi.agents.engine import AgentEngine
from mochi.config.manager import load_config


async def _run(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixtures)
    fixture_bytes = fixture_path.read_bytes()
    fixtures = load_external_qualification_fixtures(fixture_path)
    config = load_config(args.config)
    runner = ExternalQualificationRunner(engine_factory=AgentEngine)
    evidence = await runner.run(
        config=config,
        fixtures=fixtures,
        fixture_document_bytes=fixture_bytes,
        allow_external_model=bool(args.allow_external_model),
    )
    output = Path(args.output)
    if output.resolve(strict=False) in {
        fixture_path.resolve(strict=False),
        Path(args.config).resolve(strict=False),
    }:
        raise ValueError("--output must not overwrite config or fixtures")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if evidence.gate_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-external-model",
        action="store_true",
        help="Explicitly allow the configured model backend to receive bounded fixtures.",
    )
    args = parser.parse_args()
    if not args.allow_external_model:
        parser.error("--allow-external-model is required; no model was contacted")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
