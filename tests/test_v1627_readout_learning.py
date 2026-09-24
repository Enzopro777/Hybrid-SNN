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


def test_v1627_decoder_neutral_baseline_is_not_artificial_half_evidence():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    # baseline exactly equals observed activity for X and O
    lang._pool_activity_baseline["X"][:] = 1.0
    lang._pool_activity_baseline["O"][:] = 1.0
    lang._pool_activity_var["X"][:] = 1.0
    lang._pool_activity_var["O"][:] = 1.0
    lang._trial_pool_bin_counts["X"][:] = 1.0
    lang._trial_pool_bin_counts["O"][:] = 0.0
    scores = lang.readout_scores(net)
    assert scores["X"] > scores["O"]
    assert scores["O"] == 0.0
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_v1627_learning_uses_all_active_sources_but_only_eligible_pairs():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    lang._trial_source_spike_count = 6
    lang._trial_source_spike_counts = {1: 2, 2: 2, 3: 2}
    lang._trial_pool_spike_counts = {"X": 3, "O": 3, "T": 2, "A": 2, "E": 2}
    # Local eligibility: src1->X, src2->A. 29f only trains target + top rival.
    # The eligible A pair must therefore remain unchanged because A is not the top rival.
    for src, tgt, sym in [(1, 10, "X"), (2, 25, "A")]:
        lang._trial_synapse_eligibility[(src, tgt)] = 2.0
    before = dict(net.weights)
    changed = lang.apply_readout_learning(net, [1, 2, 3], "X", prediction="A")
    assert changed == 1
    assert net.weights[(1, 10)] > before[(1, 10)]
    assert net.weights[(2, 25)] == before[(2, 25)]
    assert net.weights[(3, 10)] == before[(3, 10)]
    assert net.weights[(1, 15)] == before[(1, 15)]
    audit = lang._diag_last_learning_audit
    assert audit["source_selection"] == "all_active_cortical_sources"
    assert audit["pre_post_pairs"] >= 2
    assert audit["top_rival"] == "O"


def test_v1627_does_not_transfer_credit_from_one_target_to_other_target():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    lang._trial_source_spike_count = 2
    lang._trial_source_spike_counts = {1: 2}
    lang._trial_pool_spike_counts = {"X": 2, "O": 2, "T": 2, "A": 2, "E": 2}
    lang._trial_synapse_eligibility[(1, 10)] = 2.0
    before = dict(net.weights)
    changed = lang.apply_readout_learning(net, [1], "X", prediction="O")
    assert changed == 1
    assert net.weights[(1, 10)] != before[(1, 10)]
    assert net.weights[(1, 15)] == before[(1, 15)]
    assert net.weights[(1, 20)] == before[(1, 20)]


def test_v1627_pool_homeostasis_offsets_hyperactive_pool():
    net, lang = _setup_net()
    lang.begin_clean_trial(0.0, net=net)
    lang.set_trial_frame_expectation(60)
    lang._trial_frame_count = 60
    lang._trial_pool_spike_counts = {"X": 60, "O": 15, "T": 30, "A": 30, "E": 30}
    lang._trial_pool_bin_counts["X"][:] = 5.0
    lang._trial_pool_bin_counts["O"][:] = 1.0
    lang._readout_homeostasis_trials = 8
    lang._update_pool_activity_baseline_v1627()
    assert lang._readout_pool_threshold_offset["X"] >= 0.0
    assert lang._readout_pool_threshold_offset["X"] > lang._readout_pool_threshold_offset["O"]
    assert all(v >= 0.0 for v in lang._readout_pool_threshold_offset.values())
    assert lang._readout_homeostasis_last["armed"] is True


def test_v1627_state_persists_readout_homeostasis_contract():
    net, lang = _setup_net()
    topo = lang.get_state()["readout_topology"]
    assert topo["projection_version"] == 10
    assert topo["encoder_version"].startswith("1.6.27-")
    assert topo["readout_1627"]["credit_rule"] == "1.6.29f-synapse-specific-three-factor-margin-gated"
    assert topo["readout_29f"]["version"] == "1.6.29f-homeostasis-v2"
    assert "pool_threshold_offset" in topo["readout_1627"]
