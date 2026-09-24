import numpy as np
from core.network import NeuralNetwork
from modules.language_io import LetterIOModule
from systems.competition import CompetitionSystem

def test_readout_target_bypasses_general_governor():
    net = NeuralNetwork(n=4, skip_wiring=True, max_neurons=4)
    net.readout_target_to_symbol = {2: "X"}
    net.event_queue = [(0.0, 0, 1.0, "synaptic")] * 23000
    import heapq
    heapq.heapify(net.event_queue)
    assert net.receive_spike(2, 0.01, 1.0, origin="synaptic")
    assert len(net.readout_event_queue) == 1
    assert net.event_queue_stats["readout_governor_drops"] == 0

def test_readout_learning_does_not_create_orphan_edges():
    net = NeuralNetwork(n=20, skip_wiring=True, max_neurons=20)
    lang = LetterIOModule(net=net, symbols=("X", "O"))
    lang._readout_ready = True
    lang._readout_sources = [1]
    lang._readout_pools = {"X": [2,3], "O": [4,5]}
    net.connections[1] = [2,4]
    net.weights[(1,2)] = 0.9
    net.weights[(1,4)] = 0.9
    before = set(net.weights)
    lang.net.connections[1]=[2,4]
    lang._readout_sources=[1]
    lang._readout_pools={"X":[2,3],"O":[4,5]}
    lang._readout_target_sources={2:[1],4:[1]}
    lang.begin_clean_trial(0.0, net=net)
    lang.min_learning_pool_spikes = 1; lang.min_learning_source_spikes = 1
    lang._trial_source_spike_count=1; lang._trial_source_spike_counts={1:1}
    lang.note_readout_spike(1, 10.0); lang.note_readout_spike(2, 15.0); lang.note_readout_spike(4,20.0)
    before = set(net.weights)
    changed = lang.apply_readout_learning(net, [1], "O", prediction="X")
    # 29f may also depress the true top rival; the invariant is that topology is unchanged.
    assert changed == 2
    assert set(net.weights) == before

def test_trial_reset_clears_readout_runtime_queue():
    net = NeuralNetwork(n=20, skip_wiring=True, max_neurons=20)
    lang = LetterIOModule(net=net, symbols=("X", "O"))
    lang._readout_sources = [1]
    lang._readout_pools = {"X": [2], "O": [3]}
    lang._readout_ready = True
    net.readout_target_to_symbol = {2:"X",3:"O"}
    net.receive_spike(2, 1.0, 1.0, origin="synaptic")
    assert len(net.readout_event_queue) == 1
    lang.begin_clean_trial(1.0, net=net)
    assert len(net.readout_event_queue) == 0
    assert net.membrane_potential[2] == 0.0

def test_competition_counts_are_linearized():
    net = NeuralNetwork(n=100, skip_wiring=True, max_neurons=100)
    net.neuron_region[:100] = 0
    net.active[:100] = True
    net.energy[:100] = 1.0
    system = CompetitionSystem()
    system.compete(net)
    assert system._competitive_counts[0] == 100
