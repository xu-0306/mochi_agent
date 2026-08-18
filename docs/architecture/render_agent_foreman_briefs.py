"""Render detailed implementation briefs from an Agent Foreman full plan.

The canonical JSON remains the semantic source of truth. This renderer exposes
the Main seam work and the extended worker dispatch fields that the standard
Agent Foreman Markdown renderer intentionally summarizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _lines(items: Iterable[str], *, empty: str = "None") -> list[str]:
    values = list(items)
    if not values:
        return [f"- {empty}"]
    return [f"- {value}" for value in values]


def _numbered(items: Iterable[str]) -> list[str]:
    return [f"{index}. {value}" for index, value in enumerate(items, start=1)]


def _code_lines(items: Iterable[str], *, empty: str = "None") -> list[str]:
    values = list(items)
    if not values:
        return [f"- {empty}"]
    return [f"- `{value}`" for value in values]


def _append_section(output: list[str], heading: str, body: Iterable[str]) -> None:
    output.extend([heading, "", *body, ""])


def _find_interface(plan: dict[str, Any], interface_id: str) -> dict[str, Any]:
    for interface in plan["interfaces"]:
        if interface.get("id") == interface_id:
            return interface
    raise ValueError(f"missing required interface map: {interface_id}")


def _analyze_dependency_graph(
    plan: dict[str, Any],
) -> tuple[dict[str, int], list[list[str]]]:
    nodes = plan["dependency_graph"]["nodes"]
    package_by_id = {package["id"]: package for package in plan["packages"]}
    if len(nodes) != len(set(nodes)) or set(nodes) != set(package_by_id):
        raise ValueError("dependency_graph nodes must match unique package IDs")

    predecessors = {node: [] for node in nodes}
    successors = {node: [] for node in nodes}
    for edge in plan["dependency_graph"]["edges"]:
        source = edge["from"]
        target = edge["to"]
        if source not in predecessors or target not in predecessors:
            raise ValueError(f"dependency edge references unknown package: {edge}")
        predecessors[target].append(source)
        successors[source].append(target)
    mismatched_dependencies = {
        node: {
            "package": sorted(package_by_id[node]["depends_on"]),
            "graph": sorted(predecessors[node]),
        }
        for node in nodes
        if set(package_by_id[node]["depends_on"]) != set(predecessors[node])
    }
    if mismatched_dependencies:
        raise ValueError(
            f"package depends_on and dependency_graph disagree: {mismatched_dependencies}"
        )

    node_order = {node: index for index, node in enumerate(nodes)}
    indegree = {node: len(predecessors[node]) for node in nodes}
    ready = [node for node in nodes if indegree[node] == 0]
    topological: list[str] = []
    while ready:
        ready.sort(key=node_order.__getitem__)
        node = ready.pop(0)
        topological.append(node)
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(topological) != len(nodes):
        raise ValueError("dependency_graph contains a cycle")

    earliest: dict[str, int] = {}
    longest_length: dict[str, int] = {}
    longest_paths: dict[str, list[list[str]]] = {}
    for node in topological:
        if not predecessors[node]:
            earliest[node] = 0
            longest_length[node] = 1
            longest_paths[node] = [[node]]
            continue
        earliest[node] = max(earliest[parent] for parent in predecessors[node]) + 1
        parent_length = max(longest_length[parent] for parent in predecessors[node])
        longest_length[node] = parent_length + 1
        longest_paths[node] = [
            path + [node]
            for parent in predecessors[node]
            if longest_length[parent] == parent_length
            for path in longest_paths[parent]
        ]

    graph_depth = max(longest_length.values())
    critical_paths = sorted(
        path
        for node in nodes
        if longest_length[node] == graph_depth
        for path in longest_paths[node]
    )
    return earliest, critical_paths


def _validate_schedule(plan: dict[str, Any]) -> dict[str, Any]:
    schedule = _find_interface(plan, "MAP-EXECUTION-SCHEDULE-V1")
    packages = {package["id"]: package for package in plan["packages"]}
    earliest, critical_paths = _analyze_dependency_graph(plan)

    wrong_waves = {
        package_id: {"actual": package["wave"], "expected": earliest[package_id]}
        for package_id, package in packages.items()
        if package["wave"] != earliest[package_id]
    }
    if wrong_waves:
        raise ValueError(f"package.wave is stale: {wrong_waves}")

    metrics = schedule["derived_metrics"]
    main_ids = {package_id for package_id, item in packages.items() if item["owner_role"] == "main"}
    worker_ids = set(packages) - main_ids
    release_id = "PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE"
    successors = {package_id: [] for package_id in packages}
    for edge in plan["dependency_graph"]["edges"]:
        successors[edge["from"]].append(edge["to"])
    for package_id in packages:
        if package_id == release_id:
            continue
        pending = list(successors[package_id])
        seen: set[str] = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(successors[node])
        if release_id not in seen:
            raise ValueError(f"package does not join the final release: {package_id}")
    expected_metrics = {
        "package_count": len(packages),
        "main_package_count": len(main_ids),
        "worker_package_count": len(worker_ids),
        "all_serial_unit_slots": len(packages),
        "unbounded_executor_dag_depth": max(earliest.values()) + 1,
    }
    stale_metrics = {
        key: {"actual": metrics.get(key), "expected": value}
        for key, value in expected_metrics.items()
        if metrics.get(key) != value
    }
    if stale_metrics:
        raise ValueError(f"derived schedule metrics are stale: {stale_metrics}")
    if sorted(schedule["critical_paths"]) != critical_paths:
        raise ValueError("critical_paths do not match dependency_graph")

    main_queue = schedule["main_serial_queue"]
    if len(main_queue) != len(set(main_queue)) or set(main_queue) != main_ids:
        raise ValueError("main_serial_queue must contain every Main package exactly once")

    lane_worker_ids: list[str] = []
    for lane_name, lane in schedule["parallel_lanes"].items():
        for field in ("packages", "fork_after", "cross_lane_waits", "join_at"):
            unknown = set(lane[field]) - set(packages)
            if unknown:
                raise ValueError(f"lane {lane_name} {field} references unknown packages: {unknown}")
        lane_worker_ids.extend(
            package_id
            for package_id in lane["packages"]
            if packages[package_id]["owner_role"] == "worker"
        )
    if len(lane_worker_ids) != len(set(lane_worker_ids)) or set(lane_worker_ids) != worker_ids:
        raise ValueError("parallel_lanes must cover every worker package exactly once")

    slots = schedule["recommended_unit_schedule"]
    if [slot["slot"] for slot in slots] != list(range(len(slots))):
        raise ValueError("recommended_unit_schedule slots must be contiguous and zero-based")
    capacity = schedule["capacity_assumptions"]
    package_slots: dict[str, int] = {}
    scheduled_main: list[str] = []
    for slot in slots:
        if len(slot["main"]) > capacity["main_slots"]:
            raise ValueError(f"slot {slot['slot']} exceeds Main capacity")
        if len(slot["workers"]) > capacity["worker_slots"]:
            raise ValueError(f"slot {slot['slot']} exceeds worker capacity")
        for package_id in slot["main"]:
            if package_id not in main_ids:
                raise ValueError(f"slot {slot['slot']} assigns non-Main package to Main: {package_id}")
            scheduled_main.append(package_id)
        for package_id in slot["workers"]:
            if package_id not in worker_ids:
                raise ValueError(f"slot {slot['slot']} assigns non-worker package to worker: {package_id}")
        for package_id in [*slot["main"], *slot["workers"]]:
            if package_id in package_slots:
                raise ValueError(f"package scheduled more than once: {package_id}")
            package_slots[package_id] = slot["slot"]
    if set(package_slots) != set(packages):
        raise ValueError(f"schedule package coverage mismatch: {set(packages) - set(package_slots)}")
    if scheduled_main != main_queue:
        raise ValueError("scheduled Main order does not match main_serial_queue")
    if metrics["single_main_three_worker_unit_slots"] != len(slots):
        raise ValueError("capacity schedule slot count does not match derived_metrics")
    for edge in plan["dependency_graph"]["edges"]:
        if package_slots[edge["from"]] >= package_slots[edge["to"]]:
            raise ValueError(f"schedule violates dependency edge: {edge}")
    return schedule


def _render_schedule(plan: dict[str, Any], output: list[str]) -> None:
    schedule = _validate_schedule(plan)
    metrics = schedule["derived_metrics"]
    output.extend(
        [
            "## Execution schedule",
            "",
            schedule["semantics"]["optimization_scope"],
            "",
            "| Metric | Structural value |",
            "|---|---:|",
            f"| Packages | {metrics['package_count']} |",
            f"| Main-owned packages | {metrics['main_package_count']} |",
            f"| Worker packages | {metrics['worker_package_count']} |",
            f"| All-serial unit slots | {metrics['all_serial_unit_slots']} |",
            f"| Unbounded DAG depth | {metrics['unbounded_executor_dag_depth']} |",
            "| One Main + three workers reference slots | "
            f"{metrics['single_main_three_worker_unit_slots']} |",
            "",
            f"> {metrics['external_estimate_resolution']}",
            "",
            "Package `wave` means earliest dependency eligibility. It does not authorize "
            "dispatch and does not allow concurrent Main ownership.",
            "",
            "### DAG critical paths",
            "",
            *_numbered(" → ".join(path) for path in schedule["critical_paths"]),
            "",
            "### Parallel lanes",
            "",
            "| Lane | Class | Packages | Fork after | Cross-lane waits | Join at |",
            "|---|---|---|---|---|---|",
        ]
    )
    for lane_name, lane in schedule["parallel_lanes"].items():
        output.append(
            "| "
            + " | ".join(
                [
                    lane_name,
                    lane["classification"],
                    "<br>".join(f"`{item}`" for item in lane["packages"]),
                    "<br>".join(f"`{item}`" for item in lane["fork_after"]) or "None",
                    "<br>".join(f"`{item}`" for item in lane["cross_lane_waits"]) or "None",
                    "<br>".join(f"`{item}`" for item in lane["join_at"]),
                ]
            )
            + " |"
        )
    output.extend(
        [
            "",
            "### Reference capacity schedule",
            "",
            "This is a unit-cost structural schedule, not a calendar estimate. Recompute "
            "it using actual host slots and measured durations before dispatch.",
            "",
            "| Slot | Main | Workers |",
            "|---:|---|---|",
        ]
    )
    for slot in schedule["recommended_unit_schedule"]:
        main = "<br>".join(f"`{item}`" for item in slot["main"]) or "—"
        workers = "<br>".join(f"`{item}`" for item in slot["workers"]) or "—"
        output.append(f"| {slot['slot']} | {main} | {workers} |")
    output.extend(["", "### Serialization constraints", ""])
    for constraint in schedule["serialization_constraints"]:
        packages = ", ".join(f"`{item}`" for item in constraint["packages"])
        output.append(
            f"- `{constraint['hotspot']}` — {packages}: {constraint['rule']}"
        )
    output.extend(
        [
            "",
            f"- Dispatch ready rule: {schedule['dispatch_ready_rule']}",
            f"- Join rule: {schedule['join_rule']}",
            f"- Recompute rule: {schedule['recompute_rule']}",
            "",
        ]
    )


def _render_seams(plan: dict[str, Any], output: list[str]) -> None:
    seam_map = _find_interface(plan, "MAP-SEAM-EXTRACTION")
    output.extend(
        [
            "## Main-owned Seam Extraction",
            "",
            "These extractions happen before hotspot integration. Workers must not edit "
            "the source hotspots or redefine these contracts.",
            "",
        ]
    )
    for seam_name, seam in seam_map["seams"].items():
        destinations = seam["extract_to"]
        if isinstance(destinations, str):
            destinations = [destinations]
        output.extend(
            [
                f"### {seam_name}",
                "",
                f"- Owner: `{seam['owner']}`",
                f"- Source hotspot: `{seam['source']}`",
                f"- Extract to: {', '.join(f'`{path}`' for path in destinations)}",
                f"- Integrate in: `{seam['integration']}`",
                "",
                "Source file must retain:",
                *_lines(seam["original_retains"]),
                "",
                "Source file must no longer decide:",
                *_lines(seam["must_remove"]),
                "",
            ]
        )


def _render_main_briefs(plan: dict[str, Any], output: list[str]) -> None:
    effort = _find_interface(plan, "MAP-EFFORT-SIZING")["packages"]
    briefs = _find_interface(plan, "MAP-MAIN-IMPLEMENTATION-BRIEFS")["briefs"]
    main_packages = sorted(
        (package for package in plan["packages"] if package["owner_role"] == "main"),
        key=lambda package: (package["wave"], package["id"]),
    )
    missing = [package["id"] for package in main_packages if package["id"] not in briefs]
    if missing:
        raise ValueError(f"Main packages without implementation briefs: {missing}")

    output.extend(
        [
            "## Main implementation packages",
            "",
            "Main owns contracts, lifecycle, persistence, migration, authorization, "
            "concurrency, transport, producer/consumer seams, and final integration.",
            "",
        ]
    )
    for package in main_packages:
        package_id = package["id"]
        brief = briefs[package_id]
        size = effort[package_id]["size"]
        output.extend(
            [
                f"### {package_id}",
                "",
                f"- Eligibility wave: {package['wave']} | Effort: {size}",
                f"- Depends on: {', '.join(f'`{item}`' for item in package['depends_on']) or 'None'}",
                f"- Produces: {', '.join(package['produces'])}",
                f"- Invariants: {', '.join(f'`{item}`' for item in package['invariant_ids'])}",
                "",
                "Owned paths:",
                *_code_lines(package["owned_paths"]),
                "",
                "Read first:",
                *_code_lines(package["read_first"]),
                "",
                "Never edit:",
                *_code_lines(package["prohibited_paths"]),
                "",
                "Ordered implementation:",
                *_numbered(brief["ordered_steps"]),
                "",
                "Required exit artifacts:",
                *_lines(brief["exit_artifacts"]),
                "",
                "Non-goals:",
                *_lines(brief["non_goals"]),
                "",
                "Blocking gates:",
                *_code_lines(package["verification_gate_ids"]),
                "",
            ]
        )


def _render_worker_briefs(plan: dict[str, Any], output: list[str]) -> None:
    effort = _find_interface(plan, "MAP-EFFORT-SIZING")["packages"]
    dispatch_by_id = {dispatch["package_id"]: dispatch for dispatch in plan["dispatches"]}
    worker_packages = sorted(
        (package for package in plan["packages"] if package["owner_role"] == "worker"),
        key=lambda package: (package["wave"], package["id"]),
    )
    missing = [package["id"] for package in worker_packages if package["id"] not in dispatch_by_id]
    if missing:
        raise ValueError(f"worker packages without dispatch briefs: {missing}")

    output.extend(
        [
            "## Subagent implementation briefs",
            "",
            "A package is not dispatchable until every pre-dispatch check passes and "
            "Main freezes its contracts. Model identity is unavailable, so each worker "
            "is restricted to at most one production file and one test file. Return only "
            "`implemented` or `blocked`.",
            "",
        ]
    )
    for package in worker_packages:
        package_id = package["id"]
        dispatch = dispatch_by_id[package_id]
        size = effort[package_id]["size"]
        output.extend(
            [
                f"### {package_id}",
                "",
                f"- Eligibility wave: {package['wave']} | Effort: {size}",
                f"- Objective: {dispatch['objective']}",
                f"- Depends on: {', '.join(f'`{item}`' for item in package['depends_on']) or 'None'}",
                f"- Invariants: {', '.join(f'`{item}`' for item in dispatch['invariant_ids'])}",
                f"- Gate IDs: {', '.join(f'`{item}`' for item in dispatch['gate_ids'])}",
                "",
                "Frozen contract requirements:",
                *_lines(dispatch["frozen_contracts"]),
                "",
                "Pre-dispatch checks (all must exit 0 before editing):",
                *_code_lines(dispatch["pre_dispatch_checks"]),
                "",
                "Read these entrypoints and symbols first:",
                *_code_lines(dispatch["entrypoints"]),
                "",
                "Read-only context paths:",
                *_code_lines(dispatch["read_first"]),
                "",
                "Exclusive write ownership:",
                *_code_lines(dispatch["owned_paths"]),
                "",
                "Prohibited paths:",
                *_code_lines(dispatch["prohibited_paths"]),
                "",
                "Ordered implementation:",
                *_numbered(dispatch["implementation_steps"]),
                "",
                "Must pass:",
                *_lines(dispatch["success_cases"]),
                "",
                "Must fail closed:",
                *_lines(dispatch["failure_cases"]),
                "",
                "Non-goals:",
                *_lines(dispatch["non_goals"]),
                "",
                "Focused verification commands:",
                *_code_lines(dispatch["commands"]),
                "",
                "Required artifacts:",
                *_code_lines(dispatch["expected_artifacts"]),
                "",
                "Stop conditions:",
                *_lines(dispatch["stop_conditions"]),
                "",
                "Completion checklist:",
                *_lines(dispatch["completion_checklist"]),
                "",
                "Blocked handoff template:",
                "Replace every explanatory value with exact observed evidence; never "
                "return the template text literally.",
                "```json",
                json.dumps(dispatch["blocked_report_template"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )


def render(plan_path: Path) -> str:
    source = plan_path.read_bytes()
    plan = json.loads(source.decode("utf-8"))
    if plan.get("schema_name") != "agent-foreman/plan" or plan.get("profile") != "full":
        raise ValueError("expected an agent-foreman full-profile plan")

    output = [
        "# Agent Maturity Implementation Briefs",
        "",
        "> Generated from the canonical Agent Foreman JSON. Do not edit this file by hand.",
        f"> Source: `{plan_path.as_posix()}`",
        f"> Source SHA-256: `{hashlib.sha256(source).hexdigest()}`",
        "> Regenerate: `rtk proxy python "
        "docs/architecture/render_agent_foreman_briefs.py "
        "docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json "
        "--out docs/architecture/2026-08-04-agent-maturity-implementation-briefs.md`",
        "",
        "## Dispatch authority",
        "",
        "This document prepares bounded work; it does not authorize child creation. "
        "Main must validate and freeze each consumed contract at the dispatch revision, "
        "fingerprint scope, and receive an explicit dispatch instruction before creating "
        "a subagent.",
        "",
    ]
    _render_seams(plan, output)
    _render_schedule(plan, output)
    _render_main_briefs(plan, output)
    _render_worker_briefs(plan, output)
    _append_section(output, "## Global stop conditions", _lines(plan["stop_conditions"]))
    _append_section(
        output,
        "## Fallback",
        [
            f"- Mode: `{plan['fallback_policy']['mode']}`",
            "",
            "Triggers:",
            *_lines(plan["fallback_policy"]["triggers"]),
        ],
    )
    return "\n".join(output).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(render(args.plan), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
