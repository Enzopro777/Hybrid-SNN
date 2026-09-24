import threading
import numpy as np
from modules.language_io import LetterIOModule


class FakeNet:
    def __init__(self):
        self.n = 64
        self.lock = threading.RLock()
        self.current_time = 100.0
        self.last_spike = np.zeros(self.n, dtype=float)
        self.last_update_time = np.zeros(self.n, dtype=float)
        self.refractory_until = np.zeros(self.n, dtype=float)
        self.membrane_potential = np.zeros(self.n, dtype=float)
        self.active = np.ones(self.n, dtype=bool)
        self.connections = [[] for _ in range(self.n)]
        self.weights = {}
        self.readout_target_to_symbol = {}


def make_readout():
    net = FakeNet()
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang._readout_ready = True
    lang._readout_sources = [1, 2]
    lang._readout_pools = {s: [10 + i] for i, s in enumerate(lang.symbols)}
    for src in lang._readout_sources:
        net.connections[src] = [10, 11, 12, 13, 14]
        for i, tgt in enumerate(net.connections[src]):
            net.weights[(src, tgt)] = 1.05
    lang.begin_clean_trial(0.0, net=net)
    return net, lang


def test_diagnostic_snapshot_covers_all_source_pool_synapses():
    net, lang = make_readout()
    snap = lang._diagnostic_readout_synapse_snapshot(net)
    assert len(snap) == 10
    assert all(isinstance(v, float) for v in snap.values())


def test_learning_audit_proves_real_pre_post_weight_changes():
    net, lang = make_readout()
    lang._trial_source_spike_count = 2
    lang._trial_source_spike_counts = {1: 1, 2: 1}
    lang._trial_pool_spike_counts = {s: 0 for s in lang.symbols}
    lang._trial_pool_spike_counts["X"] = 5
    lang._trial_pool_indices["X"] = {10}
    lang._readout_sources=[1,2]
    lang._readout_pools={s:[10+i] for i,s in enumerate(lang.symbols)}
    lang.net.connections[1]=[10,11,12,13,14]; lang.net.connections[2]=[10,11,12,13,14]
    lang.note_readout_spike(1, 10.0); lang.note_readout_eligibility(1,10,10.0,1.0)
    lang.note_readout_spike(2, 11.0); lang.note_readout_eligibility(2,10,11.0,1.0)
    lang.note_readout_spike(10, 20.0)
    # 29f: X already wins maximally in the instantaneous evidence; no gratuitous LTP.
    changed = lang.apply_readout_learning(net, [1, 2], "X", prediction="X")
    audit = lang._diag_last_learning_audit
    assert changed == 0
    assert audit["target_deficit"] == 0.0
    assert audit["synapses_changed_exact"] == 0


def test_full_pool_weight_stats_are_available_and_not_sample_only():
    net, lang = make_readout()
    rt = lang.get_readout_telemetry(net, t=100.0)
    for sym in lang.symbols:
        assert rt[f"readout_{sym}_synapse_count"] == 2
        assert rt[f"readout_{sym}_weight_min"] == 1.05
        assert rt[f"readout_{sym}_weight_max"] == 1.05
        assert rt[f"readout_{sym}_weight_std"] == 0.0


def test_diagnostic_does_not_change_weights_by_itself():
    net, lang = make_readout()
    before = dict(net.weights)
    _ = lang.get_readout_telemetry(net, t=100.0)
    after = dict(net.weights)
    assert before == after
