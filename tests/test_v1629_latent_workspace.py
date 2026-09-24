from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _workspace():
    from systems.latent_workspace import LatentWorkspace
    return LatentWorkspace(
        input_sources=16, hidden_dim=16, top_k=5, steps=4, seed=29,
        decoder_learning_rate=0.08, recurrent_learning_rate=0.01, context_learning_rate=0.01,
        decoder_max_update_norm=0.20, recurrent_max_update_norm=0.05, context_max_update_norm=0.05,
        stage_min_loss_gain=0.005, early_exit_enabled=True,
        early_exit_min_margin_gain=0.0015, early_exit_patience=1,
        decoder_seed_from_readout=False, neutral_decoder_scale=0.75,
        prototype_probe_enabled=True, prototype_update_rate=0.10,
    )


def test_v1629_h0_is_baseline_and_temporal_audit_exists():
    ws = _workspace()
    result = ws.run({1: [20, 2, 1], 4: [6, 0, 0], 7: [2, 1, 0]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    assert result["h0_prediction"] == result["stage_predictions"][0]
    assert result["operational_policy"] == "H0_baseline_with_utility_gated_refinement"
    assert "temporal_audit" in result
    assert result["temporal_audit"]["max_bin_share"] > 0.5


def test_v1629_neutral_decoder_does_not_seed_from_readout():
    ws = _workspace()
    ws.run({1: [2, 1, 3]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32) * 9.0)
    assert ws.decoder_seed_from_readout is False
    assert ws._decoder_seeded_from_readout is False


def test_v1629_recurrent_credit_requires_real_loss_gain():
    ws = _workspace()
    ws.run({1: [2, 1, 3], 4: [1, 2, 1]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    audit = ws.learn_from_trial("E", ("X", "O", "T", "A", "E"), apply=True)
    assert audit["decoder_update_single_pass"] is True
    if audit["learning_stage"] == "H0":
        assert audit["recurrent_credit_applied"] is False
        assert audit["recurrent_changed"] == 0


def test_v1629_eval_does_not_mutate_decoder_or_prototypes():
    ws = _workspace()
    ws.run({1: [2, 1, 3], 4: [1, 2, 1]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    ws.learn_from_trial("E", ("X", "O", "T", "A", "E"), apply=True)
    dec = ws._decoder.copy()
    protos = {k:v.copy() for k,v in ws._prototypes.items()}
    audit = ws.learn_from_trial("X", ("X", "O", "T", "A", "E"), apply=False)
    assert audit["mode"] == "eval"
    assert np.allclose(ws._decoder, dec)
    assert all(np.allclose(ws._prototypes[k], v) for k,v in protos.items())


def test_v1629_prototype_probe_persists():
    ws = _workspace()
    ws.run({1: [2, 1, 3]}, ("X", "O", "T", "A", "E"), np.ones((5, 16), dtype=np.float32))
    ws.learn_from_trial("O", ("X", "O", "T", "A", "E"), apply=True)
    assert "O" in ws._prototypes
    saved = ws.state_dict()
    ws2 = _workspace()
    ws2.restore_state(saved)
    assert "O" in ws2._prototypes
    assert np.allclose(ws._prototypes["O"], ws2._prototypes["O"])


def test_v1629_state_persists_controls():
    ws = _workspace()
    saved = ws.state_dict()
    assert saved["state_version"] >= 4
    assert saved["early_exit_enabled"] is True
    assert saved["decoder_seed_from_readout"] is False
    assert saved["stage_min_loss_gain"] == ws.stage_min_loss_gain


def test_v1629_eval_cadence_default_changed_to_five():
    from config import LATENT_WORKSPACE_EVAL_EVERY
    assert LATENT_WORKSPACE_EVAL_EVERY == 5
