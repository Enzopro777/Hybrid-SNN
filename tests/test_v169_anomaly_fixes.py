from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import threading
from modules.language_io import LetterIOModule
from modules.curriculum import VocabularyCurriculum


class FakeNet:
    def __init__(self):
        self.n = 64
        self.lock = threading.RLock()
        self.current_time = 100.0
        self.last_spike = np.zeros(self.n, dtype=float)
        self.active = np.ones(self.n, dtype=bool)


def test_detector_evidence_is_separate_from_neural_evidence_ports():
    net = FakeNet()
    lang = LetterIOModule(net=net, symbols=("X", "O"))
    lang.update_detector_evidence({"X": 0.78, "O": 0.40}, t=10.0)
    lang.update_evidence_values(net, 10.0)
    assert lang.get_detector_evidence("X") > 0.55
    assert lang.get_detector_evidence("O") == 0.0
    assert lang.input_ports["evidencia_X"].value == 0.0
    assert lang.input_ports["evidencia_O"].value == 0.0


def test_sequence_edge_events_follow_detector_transitions():
    net = FakeNet()
    lang = LetterIOModule(net=net, symbols=("X", "O"))
    lang.configure_words(["XO", "OX"])
    lang.reset_sequence_state()
    lang.update_detector_evidence({"X": 0.78, "O": 0.0}, t=100.0)
    lang.check_edge_events(net, 100.0)
    lang.update_detector_evidence({"X": 0.0, "O": 0.79}, t=500.0)
    lang.check_edge_events(net, 500.0)
    assert [s for s, _ in lang._event_log] == ["X", "O"]
    lang.update_sequence_evidence(net, 500.0, window_start=0.0, max_gap_ms=1500.0)
    assert lang.input_ports["secuencia_XO"].value > 0.0
    assert lang.input_ports["secuencia_OX"].value == 0.0


def test_curriculum_unlocks_next_letter_with_calibrated_readout_scale():
    cur = VocabularyCurriculum(["X", "O", "T"], starter_letters=["X", "O"],
                               window=8, mastery_evidence=0.40,
                               mastery_success_rate=0.70, min_trials_to_unlock=6)
    for _ in range(8):
        cur.record_trial("X", 0.44, True)
        cur.record_trial("O", 0.51, True)
    assert "T" in cur.unlocked


def test_detector_source_does_not_write_evidence_ports():
    src = (ROOT / "modules" / "detector_letras.py").read_text(encoding="utf-8")
    block_start = src.index("# v1.6.9: el score heurístico vive en un canal separado")
    block = src[block_start:block_start + 900]
    assert "update_detector_evidence" in block
    assert "input_ports.get(f\"evidencia_{_sym}\")" not in block
    assert "_port.value =" not in block
