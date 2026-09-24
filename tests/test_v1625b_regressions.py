from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_v1625b_trial_frame_count_is_defined_before_expectation_call():
    src = (ROOT / "engine" / "simulation.py").read_text()
    marker = "def training_task():"
    start = src.index(marker)
    end = src.index("self._start_training_thread(training_task", start)
    body = src[start:end]
    frames_pos = body.index("frames = 60")
    expectation_pos = body.index("lang.set_trial_frame_expectation(frames)")
    assert frames_pos < expectation_pos


def test_v1625b_temporal_core_reports_project_version():
    from core.temporal import TemporalCore
    from config import TEMPORAL_CORE_VERSION

    tc = TemporalCore()
    tc.begin_trial("letter:X:1", 100.0)
    tc.begin_frame(0, 133.333333)
    snap = tc.snapshot()
    assert tc.VERSION == "1.6.27"
    assert TEMPORAL_CORE_VERSION == "1.6.27"
    assert snap["version"] == "1.6.27"


def test_v1625b_probe_contract_exposes_temporal_core_version():
    from core.network import NeuralNetwork
    from modules.causal_probe import CausalProbe

    net = NeuralNetwork(n=200, skip_wiring=True, max_neurons=200)
    contract = CausalProbe()._runtime_contract(net)
    assert contract["project_version"] in {"IA-1.6.29", "IA-1.6.29b", "IA-1.6.29c", "IA-1.6.29d", "IA-1.6.29e", "IA-1.6.29f"}
    assert contract["temporal_core_version"] == "1.6.27"

