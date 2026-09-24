from core.network import NeuralNetwork
from modules.language_io import LetterIOModule


def _lang():
    net = NeuralNetwork(n=300, skip_wiring=True, max_neurons=300)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    net.modules["lenguaje"] = lang
    lang._readout_ready = True
    lang._readout_pools = {"X": [10], "O": [11], "T": [12], "A": [13], "E": [14]}
    lang._readout_sources = [1]
    lang._clean_trial_baseline_t = 0.0
    lang._trial_source_spike_count = 1
    lang._trial_source_indices = {1}
    return net, lang


def test_1617_pool_spike_has_single_causal_registration():
    net, lang = _lang()
    lang.note_readout_spike(11, 10.0)
    rt = lang.get_trial_readout_activity()
    assert rt["pool_spikes"] == {"X": 0, "O": 1, "T": 0, "A": 0, "E": 0}
    assert rt["pool_unique"] == {"X": 0, "O": 1, "T": 0, "A": 0, "E": 0}
    assert rt["pool_spikes_total"] == 1
    assert rt["pool_ledger_consistent"] is True


def test_1617_snapshot_is_stable_and_explicit():
    net, lang = _lang()
    lang.note_readout_spike(11, 10.0)
    snap = lang.snapshot_trial_readout()
    lang.note_readout_spike(11, 20.0)
    assert snap["pool_spikes"]["O"] == 1
    assert lang.get_last_trial_readout()["pool_spikes"]["O"] == 1
    assert lang.get_trial_readout_activity()["pool_spikes"]["O"] == 2


def test_1617_probe_end_trial_accepts_frozen_readout():
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "modules" / "causal_probe.py"
    ).read_text(encoding="utf-8")
    assert "readout_activity: Optional[dict] = None" in src
    assert "readout_pool_spikes_total" in src
    assert "readout_pool_ledger_consistent" in src
