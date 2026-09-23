from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PATTERNS = ["__pycache__", ".pytest_cache"]
FILES = {"simulation_state_meta.pkl", "simulation_state_matrices.npz", "simulation_state.pkl", "detector_calibration.json",
         "trial_commit_1.6.29b.json", "trial_commit_1.6.29b.json.tmp",
         "trial_commit_1.6.29c.json", "trial_commit_1.6.29c.json.tmp",
         "trial_commit_1.6.29d.json", "trial_commit_1.6.29d.json.tmp",
         "trial_commit_1.6.29e.json", "trial_commit_1.6.29e.json.tmp",
         "trial_commit_1.6.29f.json", "trial_commit_1.6.29f.json.tmp"}

def clean() -> int:
    removed = 0
    for p in ROOT.rglob("*"):
        if p.is_dir() and p.name in PATTERNS:
            shutil.rmtree(p, ignore_errors=True); removed += 1
    for p in ROOT.rglob("*"):
        if p.is_file() and (p.name in FILES or p.suffix == ".pyc"):
            try: p.unlink(); removed += 1
            except OSError: pass
    print(f"🧹 CLEAN EXPERIMENT: {removed} elementos eliminados.")
    print("🧠 Siguiente arranque: estado neuronal y calibración persistida = FRESH")
    return removed

if __name__ == "__main__":
    clean()
