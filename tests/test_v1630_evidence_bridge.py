from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_net():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=300, skip_wiring=True, max_neurons=300)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang._readout_ready = True
    lang._readout_sources = [1, 2, 3]
    lang._readout_pools = {s: [10+i*5] for i, s in enumerate(lang.symbols)}
    net.connections[1] = [10, 15, 20, 25, 30]
    net.connections[2] = [10, 15, 20, 25, 30]
    net.connections[3] = [10, 15, 20, 25, 30]
    for src in lang._readout_sources:
        for tgt in net.connections[src]:
            net.weights[(src, tgt)] = lang.readout_initial_weight
    lang._readout_target_sources = {t: [1, 2, 3] for t in [10, 15, 20, 25, 30]}
    return net, lang


def test_v1630_readout_uses_instantaneous_spike_evidence_and_reports_history_bias():
    from config import READOUT_HISTORY_BIAS_RATE
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    lang._pool_activity_baseline["X"][:] = 0.0
    lang._pool_activity_var["X"][:] = 1.0
    lang._trial_pool_bin_counts["X"][:] = 0.0
    lang._trial_pool_bin_counts["X"][-1] = 5.0
    lang._trial_pool_spike_counts = {"X": 5, "O": 1, "T": 1, "A": 1, "E": 1}
    lang._readout_pool_drive_ema = {"X": 100.0, "O": 10.0, "T": 10.0, "A": 10.0, "E": 10.0}
    scores = lang.readout_scores(net)
    assert scores["X"] == max(scores.values())
    audit = lang._readout_evidence_audit(scores)
    assert audit["history_bias_rate"] == READOUT_HISTORY_BIAS_RATE
    assert audit["history_bias"]["X"] > 0.0


def test_v1630_latent_bridge_context_is_measurement_only():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    lang._latent_workspace_last = {
        "h0_prediction": "T",
        "best_prediction": "T",
        "best_scores": {"X": .10, "O": .10, "T": .60, "A": .10, "E": .10},
        "accepted_scores": {"X": .10, "O": .10, "T": .55, "A": .15, "E": .10},
    }
    lang._trial_pool_bin_counts["X"][-1] = 6.0
    ctx = lang._build_bridge_context("T")
    assert ctx["latent_prediction"] == "T"
    assert ctx["latent_target_probability"] > ctx["readout_target_probability"]
    assert "history_audit" in ctx


def test_v1630_stage_reliability_persists_and_recurrent_credit_reason_is_compatible():
    from systems.latent_workspace import LatentWorkspace
    ws = LatentWorkspace(input_sources=12, hidden_dim=12, top_k=5, steps=2, seed=1630)
    ws.run({1:[3,2,4], 4:[1,2,1]}, ("X","O","T","A","E"), np.ones((5,12), dtype=np.float32))
    audit = ws.learn_from_trial("T", ("X","O","T","A","E"), apply=True, bridge_context={"h0_prediction":"T","latent_prediction":"T","readout_prediction":"X","latent_target_probability":.6,"h0_target_probability":.5,"readout_target_probability":.2})
    assert "stage_reliability_after" in audit
    saved = ws.state_dict()
    ws2 = LatentWorkspace(input_sources=12, hidden_dim=12, top_k=5, steps=2, seed=1630)
    ws2.restore_state(saved)
    assert ws2._stage_reliability == ws._stage_reliability
    assert len(ws2._bridge_failure_memory) == len(ws._bridge_failure_memory)


def test_v1630_bridge_sleep_replay_does_not_create_topology():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    before = set(net.weights)
    lang._bridge_failure_memory = [{"target":"X", "readout_prediction":"O", "latent_target_probability":.7, "readout_target_probability":.2}]
    result = lang._apply_bridge_sleep_replay()
    assert result["event_queue_touched"] is False
    assert set(net.weights) == before
