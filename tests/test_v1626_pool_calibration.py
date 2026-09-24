from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_lang(n=1800):
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=n, skip_wiring=True, max_neurons=n)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    return net, lang


def test_v1626_pool_normalization_reduces_static_pool_bias():
    net, lang = _setup_lang()
    lang.begin_clean_trial(1000.0, net=net)
    # Pool X has a high historical baseline; O has a lower baseline.
    bins = 12
    lang._pool_activity_baseline["X"][:] = 4.0
    lang._pool_activity_var["X"][:] = 0.25
    lang._pool_activity_baseline["O"][:] = 0.5
    lang._pool_activity_var["O"][:] = 0.25
    lang._trial_pool_bin_counts["X"][:] = 5.0
    lang._trial_pool_bin_counts["O"][:] = 2.0
    scores = lang.readout_scores(net, t=1400.0)
    assert scores["O"] > scores["X"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_v1626_subthreshold_delivery_creates_secondary_eligibility_only_after_real_arrival():
    net, lang = _setup_lang()
    lang.begin_clean_trial(1000.0, net=net)
    src = lang._readout_sources[0]
    tgt = next(int(t) for t in net.connections[src] if int(t) in set(lang._readout_pools["A"]))
    lang.note_readout_spike(src, 1000.0)
    lang.note_readout_eligibility(src, tgt, 1010.0, 1.0)
    assert not lang._trial_subthreshold_eligibility
    lang.note_readout_delivery(
        tgt, 1010.0, 1.0, 0.8, 0.8, 1.0, 1.55,
        refractory_blocked=False, fired=False, base_threshold=1.55,
    )
    assert (src, tgt) in lang._trial_subthreshold_eligibility
    assert lang._trial_subthreshold_eligibility[(src, tgt)] > 0.0


def test_v1626_subthreshold_bootstrap_can_train_dead_target_pool_without_spiking_it():
    net, lang = _setup_lang()
    lang.begin_clean_trial(1000.0, net=net)
    src = lang._readout_sources[0]
    tgt = next(int(t) for t in net.connections[src] if int(t) in set(lang._readout_pools["A"]))
    lang.note_readout_spike(src, 1000.0)
    lang.note_readout_eligibility(src, tgt, 1010.0, 1.0)
    lang.note_readout_delivery(
        tgt, 1010.0, 1.0, 1.0, 1.0, 1.05, 1.55,
        refractory_blocked=False, fired=False, base_threshold=1.55,
    )
    # Permitir que el ensayo sea informativo por actividad rival sin disparar el target A.
    lang._trial_source_spike_count = 2
    lang._trial_source_spike_counts = {src: 2}
    lang._trial_pool_spike_counts = {s: 2 if s == "X" else 0 for s in lang.symbols}
    before = float(net.weights[(src, tgt)])
    changed = lang.apply_readout_learning(net, [src], "A", prediction="T")
    after = float(net.weights[(src, tgt)])
    assert changed > 0
    assert after != before
    assert lang._trial_pool_spike_counts["A"] == 0


def test_v1626_competition_waits_for_normalized_dominance():
    net, lang = _setup_lang()
    lang.begin_clean_trial(1000.0, net=net)
    lang._readout_last_competition_t = -1e9
    xidx = lang._readout_pools["X"][0]
    # No normalized dominance: repeated spikes do not trigger physical competition.
    lang._compute_pool_temporal_normalized_scores = lambda: {s: 0.2 for s in lang.symbols}
    for k in range(3):
        lang.note_readout_spike(xidx, 1100.0 + k * 50.0)
    assert net.event_queue_stats.get("readout_competition_events", 0) == 0
    # Clear normalized dominance: now competition is permitted.
    lang._compute_pool_temporal_normalized_scores = lambda: {"X": 0.75, "O": 0.0625, "T": 0.0625, "A": 0.0625, "E": 0.0625}
    lang._readout_last_competition_t = -1e9
    lang.note_readout_spike(xidx, 1300.0)
    assert net.event_queue_stats.get("readout_competition_events", 0) >= 1


def test_v1626_state_persists_pool_calibration_and_version():
    net, lang = _setup_lang()
    lang._pool_baseline_trials = 7
    lang._pool_activity_baseline["X"][:] = np.arange(12, dtype=np.float32)
    state = lang.get_state()["readout_topology"]
    assert state["projection_version"] == 10
    assert state["encoder_version"].startswith("1.6.27-")
    assert state["pool_baseline_trials"] == 7
    assert len(state["pool_activity_baseline"]["X"]) == 12


def test_v1626_incomplete_trial_does_not_update_pool_baseline():
    net, lang = _setup_lang()
    lang.begin_clean_trial(1000.0, net=net)
    lang.set_trial_frame_expectation(60)
    lang._trial_frame_count = 28
    before = lang._pool_baseline_trials
    lang._trial_pool_bin_counts["A"][:] = 5.0
    lang.end_clean_trial()
    assert lang._pool_baseline_trials == before
