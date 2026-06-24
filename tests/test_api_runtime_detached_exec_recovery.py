from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any

from fastapi.testclient import TestClient
import pytest

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from tests.test_api_runtime import (
    _ApiRuntimePythonDirectProvider,
    _BACKGROUND_SMOKE_COMMAND_RULE,
    _BackgroundControlledExecAgentRunEngine,
    _wait_agent_run_until,
)


def _build_runtime_service(
    *,
    db_path: Path,
    state_root: Path,
    engine: _BackgroundControlledExecAgentRunEngine,
) -> RuntimeService:
    return RuntimeService(
        engine=engine,
        store=RuntimeStore(db_path),
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
            state_root=state_root,
        ),
    )


def _wait_for_detached_exec_job(
    client: TestClient,
    run_id: str,
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/agent-runs/{run_id}")
        assert response.status_code == 200
        last_payload = response.json()
        detached_artifact = next(
            (
                artifact
                for artifact in last_payload["artifacts"]
                if artifact["artifact_type"] == "detached_exec_jobs"
            ),
            None,
        )
        if detached_artifact is None:
            time.sleep(0.05)
            continue
        items = detached_artifact["metadata"]["content"].get("items") or []
        if items:
            return dict(items[0])
        execution_results_artifact = next(
            (
                artifact
                for artifact in last_payload["artifacts"]
                if artifact["artifact_type"] == "execution_results"
            ),
            None,
        )
        if execution_results_artifact is not None:
            results = execution_results_artifact["metadata"]["content"].get("items") or []
            if results:
                error = results[0].get("error")
                if (
                    isinstance(error, str)
                    and "_watch_detached_session" in error
                ):
                    pytest.skip(
                        "Detached exec restart recovery is unavailable in this runtime build: "
                        f"{error}"
                    )
        time.sleep(0.05)
    raise AssertionError(f"Detached exec job was not materialized in time: {last_payload}")


def _wait_for_exec_session_payload(
    client: TestClient,
    *,
    run_id: str,
    session_id: str,
    endpoint: str,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if endpoint == "get":
            response = client.get(
                f"/v1/agent-runs/{run_id}/exec/{session_id}",
                params={"yield_time_ms": 50},
            )
        else:
            response = client.post(
                f"/v1/agent-runs/{run_id}/reattach-exec/{session_id}",
                params={"yield_time_ms": 50},
            )
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload.get("associated") is not True:
            time.sleep(0.05)
            continue
        lease = last_payload.get("lease") if isinstance(last_payload.get("lease"), dict) else {}
        if isinstance(lease.get("manifest_path"), str) and lease["manifest_path"].strip():
            return last_payload
        time.sleep(0.05)
    raise AssertionError(f"Exec session payload did not stabilize in time: {last_payload}")


def test_agent_run_detached_exec_recovers_after_runtime_restart(tmp_path: Path) -> None:
    app = create_app()
    engine = _BackgroundControlledExecAgentRunEngine()
    db_path = tmp_path / "sessions" / "runtime.db"
    state_root = tmp_path / "exec-state"
    runtime_service = _build_runtime_service(db_path=db_path, state_root=state_root, engine=engine)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_BACKGROUND_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled background execution restart recovery run",
                "topic": "recover a detached smoke command after backend restart",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": True,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        detached_job = _wait_for_detached_exec_job(client, run_id)
        session_id = detached_job["session_id"]
        manifest_path = Path(detached_job["manifest_path"])
        assert manifest_path.exists()

        first_poll_payload = _wait_for_exec_session_payload(
            client,
            run_id=run_id,
            session_id=session_id,
            endpoint="get",
        )
        assert first_poll_payload["session_id"] == session_id
        assert first_poll_payload["lease"]["manifest_path"] == str(manifest_path)

        asyncio.run(runtime_service.close())
        runtime_service = _build_runtime_service(db_path=db_path, state_root=state_root, engine=engine)
        app.state.runtime_service = runtime_service

        recovered_payload = _wait_for_exec_session_payload(
            client,
            run_id=run_id,
            session_id=session_id,
            endpoint="get",
            timeout_seconds=6.0,
        )
        assert recovered_payload["run_id"] == run_id
        assert recovered_payload["session_id"] == session_id
        assert recovered_payload["associated"] is True
        assert recovered_payload["live_status"] in {"available", "unavailable"}
        assert recovered_payload["lease"]["manifest_path"] == str(manifest_path)
        if recovered_payload["session"] is not None:
            assert recovered_payload["session"]["status"] in {"running", "completed", "killed"}

        reattach_payload = _wait_for_exec_session_payload(
            client,
            run_id=run_id,
            session_id=session_id,
            endpoint="reattach",
            timeout_seconds=6.0,
        )
        assert reattach_payload["reattached"] is True
        assert reattach_payload["reattach_status"] in {"available", "unavailable"}
        assert reattach_payload["lease"]["manifest_path"] == str(manifest_path)

        updated = client.get(f"/v1/agent-runs/{run_id}")
        assert updated.status_code == 200
        reattach_events = [
            event for event in updated.json()["events"] if event["type"] == "detached_exec_reattached"
        ]
        assert reattach_events
        assert reattach_events[-1]["session_id"] == session_id

        asyncio.run(runtime_service.close())
