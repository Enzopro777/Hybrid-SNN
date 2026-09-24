import heapq

import numpy as np

from core.network import NeuralNetwork
from systems.competition import CompetitionSystem
from systems.regions import RegionSystem


def test_noncompetitive_neurons_are_preserved_by_competition():
    net = NeuralNetwork(n=20, skip_wiring=True, max_neurons=20)
    net.register_noncompetitive_neurons([0, 1, 2], role="language_infrastructure")
    net.energy[:20] = 0.01
    for r in net.regions.values():
        r.fitness = 1.0
    for _ in range(10):
        CompetitionSystem().compete(net)
    assert all(net.active[i] for i in [0, 1, 2])


def test_readout_target_uses_dedicated_queue():
    net = NeuralNetwork(n=10, skip_wiring=True, max_neurons=10)
    net.readout_target_to_symbol = {5: "X"}
    ok = net.receive_spike(5, 0.1, 1.0, origin="synaptic")
    assert ok
    assert len(net.readout_event_queue) == 1
    assert len(net.event_queue) == 0


def test_readout_queue_is_bounded():
    net = NeuralNetwork(n=4, skip_wiring=True, max_neurons=4)
    net.readout_target_to_symbol = {2: "X"}
    net.readout_queue_max = 3
    for _ in range(3):
        assert net.receive_spike(2, 0.1, 1.0, origin="synaptic")
    assert not net.receive_spike(2, 0.1, 1.0, origin="synaptic")
    assert len(net.readout_event_queue) == 3


def test_regions_count_only_competitive_neurons_for_extinction():
    net = NeuralNetwork(n=20, skip_wiring=True, max_neurons=20)
    net.register_noncompetitive_neurons(range(20), role="language_infrastructure")
    system = RegionSystem()
    system._process_competition(net)
    assert net.regions
