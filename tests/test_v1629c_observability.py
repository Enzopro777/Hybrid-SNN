from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyVision:
    def __init__(self):
        self.fovea_center_x = 0.80
        self.fovea_center_y = 0.20
        self.vel_x = 0.01
        self.vel_y = -0.02
        class Port:
            value = 0.15
        self.fovea_radius_port = Port()

    def get_fovea_telemetry(self):
        return {
            "x": self.fovea_center_x, "y": self.fovea_center_y,
            "radius": self.fovea_radius_port.value,
            "vx": self.vel_x, "vy": self.vel_y,
            "speed": (self.vel_x ** 2 + self.vel_y ** 2) ** 0.5,
        }


def test_v1629c_probe_records_fovea_trace_and_pretrial_position():
    from modules.causal_probe import CausalProbe
    probe = CausalProbe(net=None, flush_every=1000)
    vision = DummyVision()
    assert probe.begin_trial("letter:X", "X", t=100.0, expected_frames=3, vision=vision, target_xy=(0.5, 0.5))
    # Simulate three frames with movement.
    for i, (x, y) in enumerate(((0.80, 0.20), (0.60, 0.40), (0.50, 0.50))):
        vision.fovea_center_x = x
        vision.fovea_center_y = y
        probe.observe_frame(t=100.0 + (i + 1) * 33.3333333333, vision=vision)
    out = probe.end_trial(prediction="X", certainty=0.1, t=200.0, prediction_source="learned_readout", termination_reason="completed", readout_activity={"source_spikes": 3, "source_unique": 2, "pool_spikes":{"X":3,"O":0,"T":0,"A":0,"E":0}, "pool_unique":{"X":2,"O":0,"T":0,"A":0,"E":0}, "pool_spikes_total":3, "pool_ledger_consistent":True})
    assert out is not None
    trial = probe._trials[-1]
    vt = trial["vision_telemetry"]
    assert vt["fovea_start_pre_trial"]["x"] == 0.8
    assert len(vt["trace"]) == 3
    assert vt["frames_in_target"] == 1
    assert vt["path_length"] > 0.0


def test_v1629c_decision_audit_reports_authority_consistency():
    from modules.causal_probe import CausalProbe
    audit = CausalProbe._decision_audit_from_readout(
        {
            "pool_spikes": {"X":1,"O":7,"T":2,"A":0,"E":0},
            "readout_1627": {
                "pool_drive_ema": {"X":1,"O":3,"T":2,"A":0,"E":0},
                "pool_rate_ema": {"X":0.1,"O":0.3,"T":0.2,"A":0,"E":0},
                "pool_inhibition": {"X":0,"O":0,"T":0,"A":0,"E":0},
            },
            "pool_delivery_diagnostics": {
                s: {"events":1,"threshold_sum":1.3,"effective_threshold_min":1.3} for s in ("X","O","T","A","E")
            }
        },
        "O",
    )
    assert audit["pool_spike_argmax"] == "O"
    assert audit["pool_drive_argmax"] == "O"
    assert audit["argmax_matches_authority"] is True
    assert audit["prediction_path_consistent"] is True


def test_v1629c_curriculum_separates_session_and_persistent_counts():
    from modules.causal_probe import CausalProbe
    class Net:
        symbols = ("X","O","T","A","E")
        curriculum_state = {"total_trials": {"X":106,"O":106,"T":106,"A":105,"E":106}, "unlocked":["X","O","T","A","E"]}
    probe = CausalProbe(net=Net())
    for sym in ("X","O","T","A","E","X"):
        probe.begin_trial(sym, sym, t=0.0, expected_frames=0)
        trial = probe._active_trial
        trial["frames"] = 0
        probe.end_trial(prediction=sym, certainty=0.1, t=1.0, prediction_source="learned_readout", termination_reason="completed", readout_activity={"source_spikes":1,"pool_spikes":{s:(1 if s==sym else 0) for s in probe.SYMBOLS},"pool_unique":{s:(1 if s==sym else 0) for s in probe.SYMBOLS}})
    out = probe.summarize()
    assert out["curriculum"]["session_counts"]["X"] == 2
    assert out["curriculum"]["persistent_counts"]["A"] == 105


def test_v1629c_simulation_initializes_trial_complete_before_shutdown_path():
    src = (ROOT / "engine" / "simulation.py").read_text(encoding="utf-8")
    assert 'termination_reason = "incomplete_exit"\n            trial_complete = False' in src
