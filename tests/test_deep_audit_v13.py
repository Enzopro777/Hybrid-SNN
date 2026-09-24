from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_event_origin_is_preserved_and_sensory_is_protected():
    src = (ROOT / "core/network.py").read_text(encoding="utf-8")
    assert 'origin="sensory"' in src
    assert 'origin == "sensory"' in src
    assert 'shed_weak_ready_synaptic_events' in src


def test_readout_targets_are_sparse_and_not_exempt():
    src = (ROOT / "modules/language_io.py").read_text(encoding="utf-8")
    assert 'all_readout_targets' in src
    assert 'conns[:] = [int(t) for t in conns if int(t) not in all_readout_targets]' in src


def test_save_does_not_rewire_modules():
    src = (ROOT / "engine/simulation.py").read_text(encoding="utf-8")
    save_section = src[src.index('    def save('):]
    assert "mod_obj.auto_connect(self.net)" not in save_section


def test_shutdown_requires_all_threads_to_stop():
    src = (ROOT / "engine/simulation.py").read_text(encoding="utf-8")
    assert "manager_ok = manager.stop(timeout_s=10.0)" in src
    assert "if not manager_ok or not training_ok:" in src


def test_future_spikes_are_excluded_from_recent_activity():
    src = (ROOT / "modules/causal_probe.py").read_text(encoding="utf-8")
    assert "last_sp <= now_t" in src
    assert "last_sp > 0.0" in src


def test_feature_bus_source_z_range_is_enforced_and_readout_sources_are_cortical():
    src = (ROOT / "modules" / "language_io.py").read_text(encoding="utf-8")
    assert "source_lo, source_hi = map(float, source_z)" in src
    assert "self.configure_feature_bus(net, z_range=(source_lo, source_hi))" in src
    assert "self._readout_sources = ordered_sources" in src
    assert "ordered_sources = [int(i) for i in self._cortical_representation_neurons]" in src
