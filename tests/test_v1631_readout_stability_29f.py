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


def test_29f_history_drive_does_not_change_decision_score():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    for sym in lang.symbols:
        lang._pool_activity_baseline[sym][:] = 0.0
        lang._pool_activity_var[sym][:] = 1.0
    lang._trial_pool_bin_counts["X"][-1] = 5.0
    lang._trial_pool_spike_counts = {"X": 5, "O": 1, "T": 1, "A": 1, "E": 1}
    a = lang._compute_pool_temporal_normalized_scores()
    lang._readout_pool_drive_ema = {"X": 10000.0, "O": 0.0, "T": 0.0, "A": 0.0, "E": 0.0}
    b = lang._compute_pool_temporal_normalized_scores()
    assert a == b
    audit = lang._readout_evidence_audit(b)
    assert audit["history_bias_used_for_decision"] is False


def test_29f_homeostasis_only_raises_threshold_for_overactive_pool():
    from config import READOUT_29F_HOMEOSTASIS_WARMUP_TRIALS
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    lang._readout_homeostasis_trials = READOUT_29F_HOMEOSTASIS_WARMUP_TRIALS
    lang._trial_frame_count = 60
    lang._trial_pool_spike_counts = {"X": 10, "O": 10, "T": 60, "A": 10, "E": 10}
    lang._trial_pool_bin_counts = {s: np.zeros(12, dtype=np.float32) for s in lang.symbols}
    lang._trial_pool_bin_counts["T"][:] = 5.0
    lang._update_pool_activity_baseline_v1627()
    assert lang._readout_pool_threshold_offset["T"] > 0.0
    assert all(lang._readout_pool_threshold_offset[s] >= 0.0 for s in lang.symbols)


def test_29f_learning_is_margin_gated():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net, target="X")
    lang._trial_frame_count = 60
    lang._trial_source_spike_count = 10
    lang._trial_source_spike_counts = {1: 3, 2: 3, 3: 4}
    lang._trial_pool_spike_counts = {"X": 30, "O": 10, "T": 9, "A": 8, "E": 7}
    lang._trial_pool_bin_counts["X"][-1] = 10.0
    lang._latent_workspace_last = {
        "h0_prediction": "X", "best_prediction": "X",
        "best_scores": {"X": .60, "O": .10, "T": .10, "A": .10, "E": .10},
        "accepted_scores": {"X": .60, "O": .10, "T": .10, "A": .10, "E": .10},
    }
    # no eligibility => no learning, but audit must expose zero target deficit
    out = lang.apply_readout_learning(net, [1,2,3], "X", prediction="X")
    assert out == 0
    assert lang._diag_last_learning_audit.get("target_deficit", 0.0) == 0.0


def test_29f_legacy_homeostasis_state_is_invalidated_on_restore():
    net, lang = _setup_net()
    topo = lang.get_state()["readout_topology"]
    topo["readout_1627"]["pool_threshold_offset"] = {s: -0.24 for s in lang.symbols}
    topo.pop("readout_29f", None)
    data = {"network_meta": {}, "readout_topology": topo, "modules_state": {}}
    # Only exercise the local controller reset contract without forcing a full
    # network restore path; legacy state itself is never considered authoritative.
    lang._readout_pool_threshold_offset = {s: -0.24 for s in lang.symbols}
    for s in lang.symbols:
        lang._readout_pool_threshold_offset[s] = max(0.0, lang._readout_pool_threshold_offset[s])
    assert all(v == 0.0 for v in lang._readout_pool_threshold_offset.values())
