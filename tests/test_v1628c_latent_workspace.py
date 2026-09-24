from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _workspace():
    from systems.latent_workspace import LatentWorkspace
    return LatentWorkspace(
        input_sources=16, hidden_dim=16, top_k=5, steps=4, seed=28,
        decoder_learning_rate=0.08, recurrent_learning_rate=0.01, context_learning_rate=0.01,
        decoder_max_update_norm=0.15, recurrent_max_update_norm=0.05, context_max_update_norm=0.05,
    )


def test_v1628c_records_ablation_and_stage_utility():
    ws = _workspace()
    result = ws.run({1: [2, 1, 3], 4: [1, 2, 1], 7: [3, 1, 0]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    assert set(("C-only", "S-only", "C+S", "recurrent")) <= set(result["ablation"])
    assert "H0" in result["stage_diagnostics"]
    assert len(result["stage_diagnostics"]) >= 2
    assert "innovation_sequence" in result
    assert result["halt_reason"] in {"max_steps", "stable_state_or_utility", "h0_only"}


def test_v1628c_decoder_update_is_single_aggregated_pass_and_bounded():
    ws = _workspace()
    ws.run({1: [2, 1, 3], 4: [1, 2, 1]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    before = ws._decoder.copy()
    audit = ws.learn_from_trial("E", ("X", "O", "T", "A", "E"), apply=True)
    delta_norm = float(np.linalg.norm(ws._decoder - before))
    assert audit["learning_applied"] is True
    assert audit["decoder_update_single_pass"] is True
    assert audit["recurrent_credit_uses_pre_update_decoder"] is True
    assert delta_norm <= ws.decoder_max_update_norm + 1e-6
    assert audit["decoder_changed"] > 0


def test_v1628c_eval_does_not_change_parameters():
    ws = _workspace()
    ws.run({1: [2, 1, 3], 4: [1, 2, 1]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    dec = ws._decoder.copy(); rec = ws._recurrent.copy(); ctx = ws._context_projection.copy(); steps = ws.learning_steps
    audit = ws.learn_from_trial("E", ("X", "O", "T", "A", "E"), apply=False)
    assert audit["mode"] == "eval"
    assert audit["learning_applied"] is False
    assert np.allclose(ws._decoder, dec)
    assert np.allclose(ws._recurrent, rec)
    assert np.allclose(ws._context_projection, ctx)
    assert ws.learning_steps == steps


def test_v1628c_state_persists_new_learning_limits():
    ws = _workspace()
    ws.run({1: [2, 1, 3]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    ws.learn_from_trial("O", ("X", "O", "T", "A", "E"))
    saved = ws.state_dict()
    ws2 = _workspace()
    ws2.restore_state(saved)
    assert ws2.STATE_VERSION >= 4
    assert ws2.decoder_max_update_norm == ws.decoder_max_update_norm
    assert np.allclose(ws2._decoder, ws._decoder)


def test_v1628c_incomplete_trial_still_cannot_learn():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1200, skip_wiring=True, max_neurons=1200)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    lang.begin_clean_trial(1000.0, net=net)
    lang.set_trial_frame_expectation(60)
    src = lang._readout_sources[0]
    lang._trial_source_bin_counts = {src: [1, 2, 1]}
    lang._trial_source_spike_count = 4
    lang.run_latent_workspace_reasoning(net)
    before = lang._latent_workspace._decoder.copy()
    audit = lang.learn_latent_workspace_trial("X", completed=False)
    assert audit["learning_applied"] is False
    assert audit["skip_reason"] == "incomplete_trial"
    assert np.allclose(before, lang._latent_workspace._decoder)


def test_v1628c_queue_audit_is_observational():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1200, skip_wiring=True, max_neurons=1200)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    lang.begin_clean_trial(1000.0, net=net)
    audit_before = dict(lang._trial_queue_before)
    src = lang._readout_sources[0]
    lang._trial_source_bin_counts = {src: [1, 2, 1]}
    lang._trial_source_spike_count = 4
    lang.run_latent_workspace_reasoning(net)
    lang.learn_latent_workspace_trial("X", completed=True)
    snapshot = lang.get_trial_readout_activity()
    assert snapshot["queue_trial_audit"]["before"] == audit_before
    assert snapshot["queue_trial_audit"]["delta"]["enqueued_total"] == 0
    assert snapshot["queue_trial_audit"]["delta"]["processed_total"] == 0


def test_v1628c_eval_cadence_is_deterministic():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1200, skip_wiring=True, max_neurons=1200)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    lang._cortical_trials_completed = 19
    lang.begin_clean_trial(1000.0, net=net)
    assert lang._latent_trial_sequence == 20
    assert lang._latent_trial_mode == "eval"
