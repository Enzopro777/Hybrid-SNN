from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1200, skip_wiring=True, max_neurons=1200)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    lang.begin_clean_trial(1000.0, net=net)
    lang.set_trial_frame_expectation(60)
    return net, lang


def test_v1628b_temporal_flatten_is_bin_major():
    from systems.latent_workspace import LatentWorkspace
    ws = LatentWorkspace(input_sources=4, temporal_bins=3, hidden_dim=8, top_k=4, seed=1)
    x = ws._build_temporal_tensor({1: [1, 2, 3], 3: [4, 5, 6]})
    assert x[0 * 4 + 1] > 0
    assert x[1 * 4 + 1] > 0
    assert x[2 * 4 + 1] > 0
    assert x[0 * 4 + 3] > 0
    assert x[1 * 4 + 3] > 0
    assert x[2 * 4 + 3] > 0
    assert np.count_nonzero(x) == 6


def test_v1628b_source_bins_use_expected_duration_not_observed_frame_count():
    net, lang = _setup()
    src = lang._readout_sources[0]
    # Con expected_frames=60, 800 ms cae en el bin central. En 1.6.28 el
    # denominador podía usar los frames observados y desplazarlo al último bin.
    lang.note_readout_spike(src, 1800.0)
    vals = lang._trial_source_bin_counts[src]
    assert vals == [0, 1, 0]


def test_v1628b_s_and_h_are_independent_states():
    from systems.latent_workspace import LatentWorkspace
    ws = LatentWorkspace(input_sources=12, hidden_dim=12, top_k=4, steps=4, seed=8)
    decoder_seed = np.ones((5, 12), dtype=np.float32)
    result = ws.run({0: [1, 2, 3], 4: [2, 0, 1], 7: [1, 1, 1]}, ("X", "O", "T", "A", "E"), decoder_seed)
    assert result["active"] is True
    assert np.count_nonzero(ws.context_state) > 0
    assert np.count_nonzero(ws.workspace_state) > 0
    assert not np.allclose(ws.context_state, ws.workspace_state)
    assert np.allclose(ws.last_states[0], ws.last_states[0])


def test_v1628b_decoder_persists_and_learning_changes_parameters():
    from systems.latent_workspace import LatentWorkspace
    ws = LatentWorkspace(input_sources=16, hidden_dim=16, top_k=5, steps=4, seed=10,
                         decoder_learning_rate=0.08, recurrent_learning_rate=0.01,
                         context_learning_rate=0.01)
    rmat = np.ones((5, 16), dtype=np.float32)
    ws.run({1: [2, 1, 3], 4: [1, 2, 1]}, ("X", "O", "T", "A", "E"), rmat)
    before = ws._decoder.copy()
    audit = ws.learn_from_trial("E", ("X", "O", "T", "A", "E"))
    after = ws._decoder.copy()
    assert audit["learning_applied"] is True
    assert audit["decoder_changed"] > 0
    assert float(np.sum(np.abs(after - before))) > 0.0
    saved = ws.state_dict()
    ws2 = LatentWorkspace(input_sources=16, hidden_dim=16, top_k=5, steps=4, seed=999)
    ws2.restore_state(saved)
    assert np.allclose(ws2._decoder, ws._decoder)
    assert np.allclose(ws2._recurrent, ws._recurrent)
    assert ws2.learning_steps == ws.learning_steps


def test_v1628b_incomplete_trial_does_not_learn():
    net, lang = _setup()
    src = lang._readout_sources[0]
    lang._trial_source_bin_counts = {src: [2, 1, 1]}
    lang._trial_source_spike_count = 4
    lang.run_latent_workspace_reasoning(net)
    before = lang._latent_workspace._decoder.copy()
    audit = lang.learn_latent_workspace_trial("O", completed=False)
    after = lang._latent_workspace._decoder.copy()
    assert audit["learning_applied"] is False
    assert audit["skip_reason"] == "incomplete_trial"
    assert np.allclose(before, after)


def test_v1628b_best_stage_is_explicit_and_authority_safe():
    from systems.latent_workspace import LatentWorkspace
    ws = LatentWorkspace(input_sources=10, hidden_dim=10, top_k=4, steps=4, seed=17)
    r = ws.run({0: [3, 2, 1], 2: [0, 2, 4], 5: [1, 3, 1]}, ("X", "O", "T", "A", "E"), np.ones((5, 10), dtype=np.float32))
    assert r["best_stage"] in r["stage_scores"]
    assert r["best_prediction"] == r["stage_predictions"][list(r["stage_scores"].keys()).index(r["best_stage"])]
    assert r["authoritative"] is False


def test_v1628b_probe_records_termination_reason():
    from modules.causal_probe import CausalProbe
    net, _ = _setup()
    probe = CausalProbe(net=net)
    assert probe.begin_trial("letter:T", ground_truth="T", t=1000.0, expected_frames=60)
    result = probe.end_trial(prediction=None, certainty=0.0, t=1133.0,
                             prediction_source="learned_readout_inactive",
                             termination_reason="simulation_shutdown")
    assert result["frames_observed"] == 0
    assert result["frames_complete"] is False
    saved = probe.summarize()["trials"]["recent"][-1]
    assert saved["termination_reason"] == "simulation_shutdown"
    assert saved["termination_frame"] == 0
    assert probe.summarize()["metrics"]["trials_completed"] == 0
    assert probe.summarize()["metrics"]["trials_recorded"] == 1


def test_v1628b_language_latent_learning_is_event_free():
    net, lang = _setup()
    src = lang._readout_sources[0]
    lang._trial_source_bin_counts = {src: [2, 1, 2]}
    lang._trial_source_spike_count = 5
    before_events = int(net.event_queue_stats.get("enqueued_total", 0))
    result = lang.run_latent_workspace_reasoning(net)
    learning = lang.learn_latent_workspace_trial("X", completed=True)
    after_events = int(net.event_queue_stats.get("enqueued_total", 0))
    assert result["active"] is True
    assert learning["learning_applied"] is True
    assert before_events == after_events
