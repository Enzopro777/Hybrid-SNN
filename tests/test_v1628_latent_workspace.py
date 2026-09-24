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
    return net, lang


def test_v1628_latent_workspace_is_deterministic_and_bounded():
    from systems.latent_workspace import LatentWorkspace
    ws1 = LatentWorkspace(input_sources=12, temporal_bins=3, hidden_dim=12, steps=4, seed=123)
    ws2 = LatentWorkspace(input_sources=12, temporal_bins=3, hidden_dim=12, steps=4, seed=123)
    bins = {0: [1, 2, 3], 2: [0, 3, 1], 7: [4, 1, 0]}
    rmat = np.ones((5, 12), dtype=np.float32)
    a = ws1.run(bins, ("X", "O", "T", "A", "E"), rmat)
    b = ws2.run(bins, ("X", "O", "T", "A", "E"), rmat)
    assert a["stage_predictions"] == b["stage_predictions"]
    assert np.isfinite(ws1.workspace_state).all()
    assert float(np.max(ws1.workspace_state)) <= 1.0 + 1e-6
    assert int(a["workspace_nonzero"]) <= 24


def test_v1628_empty_workspace_does_not_fabricate_probabilities():
    from systems.latent_workspace import LatentWorkspace
    ws = LatentWorkspace(input_sources=8, hidden_dim=8, top_k=4, steps=4, seed=321)
    rmat = np.ones((5, 8), dtype=np.float32)
    r = ws.run({}, ("X", "O", "T", "A", "E"), rmat)
    assert r["active"] is False
    assert all(float(v) == 0.0 for v in r["stage_scores"]["H0"].values())
    assert r["stage_predictions"] == [None]


def test_v1628_language_workspace_uses_cortical_bins_only_and_is_observational():
    net, lang = _setup()
    lang._trial_source_bin_counts = {lang._readout_sources[0]: [2, 1, 3], lang._readout_sources[1]: [1, 0, 2]}
    lang._trial_source_spike_count = 9
    before_events = int(net.event_queue_stats.get("enqueued_total", 0))
    result = lang.run_latent_workspace_reasoning(net)
    after_events = int(net.event_queue_stats.get("enqueued_total", 0))
    assert result["active"] is True
    assert result["authoritative"] is False
    assert before_events == after_events
    assert result["input_source_spikes"] == 9
    assert 2 <= len(result["stage_predictions"]) <= 5


def test_v1628_workspace_state_is_in_readout_ledger():
    net, lang = _setup()
    lang._trial_source_bin_counts = {lang._readout_sources[0]: [2, 1, 3]}
    lang.run_latent_workspace_reasoning(net)
    activity = lang.get_trial_readout_activity()
    assert "latent_workspace" in activity
    assert activity["latent_workspace"]["version"].startswith(("1.6.28b-", "1.6.28c-", "1.6.29-", "1.6.29b-", "1.6.29d-", "1.6.29e-", "1.6.29f-"))


def test_v1628_state_persists_workspace_configuration():
    net, lang = _setup()
    lang._trial_source_bin_counts = {lang._readout_sources[0]: [1, 1, 1]}
    lang.run_latent_workspace_reasoning(net)
    topo = lang.get_state()["readout_topology"]
    assert topo["projection_version"] == 10
    assert topo["latent_workspace_version"].startswith(("1.6.29-latent-workspace-v4", "1.6.29b-latent-workspace-v5", "1.6.29e-latent-workspace-v7", "1.6.29f-latent-workspace-v7"))
    assert topo["latent_workspace_config"]["steps"] == 4
    assert topo["latent_workspace_config"]["decoder_learning_rate"] > 0.0
    assert topo["latent_workspace_state"]
