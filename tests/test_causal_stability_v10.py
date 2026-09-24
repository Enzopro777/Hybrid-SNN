import numpy as np


def test_region_energy_clamps_without_negative_publication():
    from core.region import Region
    r = Region(0)
    r.energy = 0.12
    r.update_energy(-5.0)
    assert r.energy >= 0.1


def test_region_adjust_energy_clamps():
    from core.region import Region
    r = Region(0)
    r.energy = 0.2
    r.adjust_energy(-5.0)
    assert r.energy >= 0.1


def test_queue_stats_expose_pressure_metrics():
    from core.network import NeuralNetwork
    net = NeuralNetwork(n=10, skip_wiring=True, max_neurons=10)
    stats = net.get_event_queue_stats()
    assert "production_minus_consumption" in stats
    assert "governor_drop_rate" in stats
    assert "ready_backlog_peak" in stats


def test_probe_reports_region_energy_separately():
    from modules.causal_probe import CausalProbe

    class R:
        def __init__(self, e): self.energy = e

    class Net:
        n = 1; max_neurons = 1; current_time = 0.0; modules = {}
        regions = {0: R(-0.1)}
        active = np.array([True])
        energy = np.array([1.0])
        def get_event_queue_stats(self):
            return {"queue_size": 0, "ready_count": 0, "future_count": 0}

    summary = CausalProbe(net=Net()).summarize()
    assert summary["health"]["region_energy"]["negative_count"] == 1


def test_probe_exposes_execution_identity():
    from modules.causal_probe import CausalProbe

    summary = CausalProbe().summarize()
    ident = summary["runtime_contract"]["execution_identity"]
    assert "workspace_root" in ident
    assert "declared_project_version" in ident
    assert "version_label_consistent" in ident
