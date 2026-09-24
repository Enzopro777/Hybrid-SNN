from pathlib import Path
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _workspace():
    from systems.latent_workspace import LatentWorkspace
    return LatentWorkspace(
        input_sources=16, hidden_dim=16, top_k=5, steps=4, seed=129,
        decoder_learning_rate=0.08, recurrent_learning_rate=0.01, context_learning_rate=0.01,
        decoder_max_update_norm=0.45, recurrent_max_update_norm=0.05, context_max_update_norm=0.05,
        stage_min_loss_gain=0.005, early_exit_enabled=True, early_exit_min_margin_gain=0.0015,
        early_exit_patience=1, decoder_seed_from_readout=False, neutral_decoder_scale=0.75,
        prototype_probe_enabled=True, prototype_update_rate=0.08, replay_capacity=5,
        replay_per_class=1, replay_weight=0.15, temporal_shadow_enabled=True,
        temporal_shadow_learning_rate=0.02, temporal_shadow_max_update_norm=0.25,
    )


def test_v1629b_h0_remains_operational_baseline():
    ws = _workspace()
    result = ws.run({1: [5, 5, 5], 4: [2, 3, 4]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    assert result["accepted_stage"] == "H0"
    assert result["accepted_prediction"] == result["h0_prediction"]
    assert result["operational_policy"] == "H0_baseline_with_utility_gated_refinement"


def test_v1629b_temporal_shadow_is_non_authoritative_and_learnable():
    ws = _workspace()
    source = {1: [8, 6, 7], 2: [5, 4, 3]}
    result = ws.run(source, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    assert result["temporal_shadow"]["enabled"] is True
    assert result["temporal_shadow"]["available"] is True
    before = ws._temporal_shadow_decoder.copy()
    audit = ws.learn_from_trial("X", ("X", "O", "T", "A", "E"), apply=True)
    assert audit["temporal_shadow"]["applied"] is True
    assert not np.allclose(before, ws._temporal_shadow_decoder)


def test_v1629b_replay_memory_stays_small_and_is_used():
    ws = _workspace()
    symbols = ("X", "O", "T", "A", "E")
    for label in symbols:
        ws.run({1: [2, 3, 4], 3: [1, 2, 2]}, symbols, np.ones((5, 16), dtype=np.float32))
        ws.learn_from_trial(label, symbols, apply=True)
    assert all(len(v) <= 5 for v in ws._prototype_memory.values())
    ws.run({1: [4, 3, 5]}, symbols, np.ones((5, 16), dtype=np.float32))
    audit = ws.learn_from_trial("X", symbols, apply=True)
    assert audit["decoder_replay_examples"] >= 1


def test_v1629b_eval_does_not_mutate_shadow_or_replay():
    ws = _workspace()
    symbols = ("X", "O", "T", "A", "E")
    ws.run({1: [3, 2, 4]}, symbols, np.ones((5, 16), dtype=np.float32))
    ws.learn_from_trial("O", symbols, apply=True)
    before_shadow = ws._temporal_shadow_decoder.copy()
    before_memory = {k:[x.copy() for x in v] for k,v in ws._prototype_memory.items()}
    ws.run({1: [2, 1, 4]}, symbols, np.ones((5, 16), dtype=np.float32))
    audit = ws.learn_from_trial("X", symbols, apply=False)
    assert audit["mode"] == "eval"
    assert np.allclose(before_shadow, ws._temporal_shadow_decoder)
    assert all(np.allclose(before_memory[k][i], v[i]) for k,v in before_memory.items() for i in range(len(v)))


def test_v1629b_state_roundtrip_keeps_replay_and_shadow():
    ws = _workspace()
    symbols = ("X", "O", "T", "A", "E")
    ws.run({1: [5, 4, 3]}, symbols, np.ones((5, 16), dtype=np.float32))
    ws.learn_from_trial("T", symbols, apply=True)
    saved = ws.state_dict()
    assert saved["state_version"] >= 5
    ws2 = _workspace()
    ws2.restore_state(saved)
    assert "T" in ws2._prototype_memory
    assert ws2._temporal_shadow_decoder is not None
    assert np.allclose(ws._temporal_shadow_decoder, ws2._temporal_shadow_decoder)


def test_v1629b_trial_commit_journal_is_atomic_and_validates():
    from systems.trial_commit_journal import TrialCommitJournal
    path = ROOT / "_test_trial_commit_journal.json"
    try:
        j = TrialCommitJournal(str(path))
        data = j.commit(trial_id="letter:X:5", sequence=5, label="X", current_time=123.0, frames=60, completed=True, latent_summary={"h0_prediction":"X"})
        loaded = j.load()
        assert loaded is not None
        assert loaded["sequence"] == 5
        assert loaded["commit_digest"] == data["commit_digest"]
    finally:
        if path.exists():
            path.unlink()
