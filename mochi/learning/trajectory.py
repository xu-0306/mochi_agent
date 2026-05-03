"""軌跡記錄器 — Phase 5A 無 LLM 依賴實作。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from mochi.learning.types import Trajectory, TrajectoryOutcome, TrajectoryStep

VALID_OUTCOMES: set[str] = {"success", "failure", "partial", "unknown"}


class TrajectoryLogger:
    """嵌入 ReAct Loop 自動記錄每個步驟。"""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._trajectories: dict[str, Trajectory] = {}

    def start(self, task: str) -> str:
        """開始記錄新軌跡，回傳 trajectory_id。"""
        trajectory_id = str(uuid4())
        self._trajectories[trajectory_id] = Trajectory(
            trajectory_id=trajectory_id,
            task_description=task,
            steps=[],
            outcome="unknown",
        )
        return trajectory_id

    def log_step(self, trajectory_id: str, step: TrajectoryStep) -> None:
        """記錄一個執行步驟。"""
        trajectory = self._get(trajectory_id)
        trajectory.steps.append(step)
        trajectory.total_tokens += step.tokens_used
        trajectory.total_duration_ms += step.duration_ms

    def finish(self, trajectory_id: str, outcome: str, feedback: str | None = None) -> None:
        """結束軌跡記錄。"""
        if outcome not in VALID_OUTCOMES:
            valid = ", ".join(sorted(VALID_OUTCOMES))
            raise ValueError(f"Invalid trajectory outcome: {outcome!r}. Expected one of: {valid}")

        trajectory = self._get(trajectory_id)
        trajectory.outcome = cast(TrajectoryOutcome, outcome)
        trajectory.user_feedback = feedback

        if self.storage_path is not None:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.storage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")

    def export(self, trajectory_id: str) -> Trajectory:
        """匯出完整軌跡。"""
        return Trajectory.from_dict(self._get(trajectory_id).to_dict())

    def load_all(self) -> list[Trajectory]:
        """從 JSONL 儲存檔載入所有已完成軌跡。"""
        if self.storage_path is None or not self.storage_path.exists():
            return []

        trajectories: list[Trajectory] = []
        with self.storage_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trajectories.append(Trajectory.from_dict(json.loads(line)))
        return trajectories

    def load(self, trajectory_id: str) -> Trajectory:
        """從 JSONL 儲存檔載入指定軌跡。"""
        for trajectory in self.load_all():
            if trajectory.trajectory_id == trajectory_id:
                return trajectory
        raise KeyError(f"Unknown trajectory_id: {trajectory_id}")

    def _get(self, trajectory_id: str) -> Trajectory:
        try:
            return self._trajectories[trajectory_id]
        except KeyError as exc:
            raise KeyError(f"Unknown trajectory_id: {trajectory_id}") from exc
