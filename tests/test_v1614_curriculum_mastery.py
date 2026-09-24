from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_mastery_requires_readout_separation_margin():
    from modules.curriculum import VocabularyCurriculum
    c = VocabularyCurriculum(["X", "O"], starter_letters=["X", "O"], window=8,
                             mastery_evidence=0.40, mastery_success_rate=0.70,
                             mastery_margin=0.10)
    for _ in range(8):
        c.record_trial("X", 0.45, True, margin=0.04)
        c.record_trial("O", 0.45, True, margin=0.20)
    assert not c.is_mastered("X")
    assert c.is_mastered("O")


def test_old_two_field_history_remains_backward_compatible_but_not_mastered_under_margin_rule():
    from modules.curriculum import VocabularyCurriculum
    c = VocabularyCurriculum(["X"], starter_letters=["X"], window=8,
                             mastery_evidence=0.40, mastery_success_rate=0.70,
                             mastery_margin=0.10)
    c.load_dict({"vocabulary": ["X"], "unlocked": ["X"],
                 "history": {"X": [(0.45, True)] * 8},
                 "total_trials": {"X": 8}})
    assert c.mastery("X") == (0.45, 1.0)
    assert c.mastery_detail("X")[2] == 0.0
    assert not c.is_mastered("X")


def test_status_exposes_margin_so_missing_green_is_auditable():
    from modules.curriculum import VocabularyCurriculum
    c = VocabularyCurriculum(["X"], starter_letters=["X"], window=8, mastery_margin=0.10)
    c.record_trial("X", 0.20, True, margin=0.03)
    status = c.status()
    assert "X📚" in status
    assert "m=0.03" in status


def test_causal_probe_brand_matches_1614():
    src = (ROOT / "modules" / "causal_probe.py").read_text(encoding="utf-8")
    assert ('PROJECT_VERSION = "IA-1.6.29"' in src) or ('PROJECT_VERSION = "IA-1.6.29c"' in src) or ('PROJECT_VERSION = "IA-1.6.29d"' in src) or ('PROJECT_VERSION = "IA-1.6.29f"' in src)
