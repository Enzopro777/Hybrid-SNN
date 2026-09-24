"""IA 1.6.29b - durable trial-boundary journal.

Registra atómicamente la última frontera de trial comprometida. No guarda la
red completa mientras los hilos biológicos están activos; el checkpoint
completo sigue siendo SimulationEngine.save() al cierre ordenado.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


class TrialCommitJournal:
    VERSION = "1.6.29c-trial-commit-journal-v1"

    def __init__(self, path: str = "trial_commit_1.6.29c.json") -> None:
        self.path = Path(path)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): TrialCommitJournal._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrialCommitJournal._json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, float):
            return value if value == value and abs(value) != float("inf") else 0.0
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def digest(payload: Dict[str, Any]) -> str:
        raw = json.dumps(TrialCommitJournal._json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def commit(
        self,
        *,
        trial_id: str,
        sequence: int,
        label: str,
        current_time: float,
        frames: int,
        completed: bool,
        trial_mode: str = "train",
        learning_steps: int = 0,
        latent_summary: Optional[Dict[str, Any]] = None,
        readout_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "journal_version": self.VERSION,
            "project": "IA-1.6.29c",
            "committed": bool(completed),
            "trial_id": str(trial_id),
            "sequence": int(sequence),
            "label": str(label),
            "current_time": float(current_time),
            "frames": int(frames),
            "trial_mode": str(trial_mode),
            "learning_steps": int(learning_steps),
            "latent_summary": latent_summary or {},
            "readout_summary": readout_summary or {},
        }
        envelope = dict(payload)
        envelope["commit_digest"] = self.digest(payload)
        envelope["write_pid"] = int(os.getpid())
        envelope["committed_at_unix"] = __import__("time").time()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(envelope, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
        return envelope

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            digest = data.get("commit_digest")
            core = {k: v for k, v in data.items() if k not in {"commit_digest", "write_pid", "committed_at_unix"}}
            if digest and digest == self.digest(core):
                return data
        except Exception:
            return None
        return None
