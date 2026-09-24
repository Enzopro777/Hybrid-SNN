from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_causal_probe_summarize_initializes_health_before_use():
    from modules.causal_probe import CausalProbe

    class FakeNet:
        n = 2
        max_neurons = 2
        current_time = 0.0
        modules = {}
        active = np.array([True, True])
        energy = np.array([1.0, 0.5])

        def get_event_queue_stats(self):
            return {"queue_size": 0, "ready_count": 0, "future_count": 0}

    probe = CausalProbe(net=FakeNet())
    summary = probe.summarize()
    assert summary["health"]["status"] == "ok"
    assert summary["health"]["event_queue"]["queue_size"] == 0
    assert summary["telemetry_integrity"]["status"] == "ok"


def test_trial_summary_separates_prediction_match_from_readout_activity():
    from modules.causal_probe import CausalProbe

    probe = CausalProbe()
    trial = {
        "id": 1,
        "ground_truth": "X",
        "prediction": "X",
        "prediction_source": "learned_readout",
        "frames": 60,
        "certainty": 0.8,
        "evidence_peak": {"X": 0.5, "O": 0.0},
        "evidence_first_t": {"X": 1.0, "O": None},
        "readout_sources_fired_sum": 0,
        "readout_pool_spikes": {"X": 0, "O": 0},
    }
    s = probe._trial_causal_summary(trial)
    assert s["prediction_match"] is True
    assert s["learned_readout_active"] is False
    assert s["learned_readout_valid"] is False


def test_source_manifest_version_is_v09():
    src = (ROOT / "modules" / "causal_probe.py").read_text(encoding="utf-8")
    assert 'VERSION = "causal-probe-v1.5"' in src
    assert '"telemetry_integrity"' in src

def test_flush_writes_snapshot_without_queue_health_unbound(tmp_path):
    from modules.causal_probe import CausalProbe

    class FakeNet:
        n = 1
        max_neurons = 1
        current_time = 1.0
        modules = {}
        active = np.array([True])
        energy = np.array([1.0])

        def get_event_queue_stats(self):
            return {"queue_size": 0, "ready_count": 0, "future_count": 0}

    out = tmp_path / "probe.json"
    probe = CausalProbe(net=FakeNet(), output_path=str(out))
    assert probe.flush()
    text = out.read_text(encoding="utf-8")
    assert '"telemetry_integrity"' in text
    assert '"status": "ok"' in text
