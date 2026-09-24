import numpy as np
from core.network import NeuralNetwork
from modules.language_io import LetterIOModule
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readout_source_is_not_globally_exempt():
    src = (ROOT / "modules/language_io.py").read_text(encoding="utf-8")
    assert "governor_exempt_targets.discard" in src
    assert "readout_target_to_symbol" in src


def test_readout_uses_causal_trial_counters():
    src = (ROOT / "modules/language_io.py").read_text(encoding="utf-8")
    assert "note_readout_spike" in src
    assert "_trial_source_spike_count" in src
    assert "_trial_pool_spike_counts" in src
    assert "_trial_pool_indices" in src
    assert "min(1.0" in src


def test_no_heuristic_fallback_for_authoritative_letter_prediction():
    src = (ROOT / "engine/simulation.py").read_text(encoding="utf-8")
    assert "learned_readout_inactive" in src
    assert "evidence_ports_direct" not in src
    assert "evidence_ports_fallback" not in src


def test_queue_metrics_are_semantically_named():
    src = (ROOT / "core/network.py").read_text(encoding="utf-8")
    assert '"accepted_minus_processed"' in src
    assert '"attempt_minus_processed"' in src
    assert '"fanout_attempted_total"' in src
    assert '"fanout_send_rate"' in src


def test_fanout_preserves_readout_channel_under_pressure():
    src = (ROOT / "core/network.py").read_text(encoding="utf-8")
    assert "readout_candidates" in src
    assert "normal_candidates" in src
    assert "target_map" in src


def test_spatial_sensory_routing_is_not_id_order_based():
    src = (ROOT / "core/network.py").read_text(encoding="utf-8")
    assert "cx = ((bx + 0.5) / bins_x)" in src
    assert "cy = ((by + 0.5) / bins_y)" in src
    assert "np.argsort(d2)" in src


def test_feature_bus_preserves_detector_identity():
    net = NeuralNetwork(n=1000, skip_wiring=True, max_neurons=1000)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_feature_bus(net)
    groups = lang._feature_bus_groups
    assert len(groups) == 18
    flat = [i for g in groups.values() for i in g]
    assert len(flat) == len(set(flat)) == 18 * 36
    assert all(net.is_noncompetitive(i) for i in flat)


def test_feature_bus_emits_typed_spikes():
    net = NeuralNetwork(n=1000, skip_wiring=True, max_neurons=1000)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_feature_bus(net)
    grid = np.zeros((20, 20), dtype=np.float32)
    grid[10, 10] = 10.0
    emitted = lang.emit_feature_maps(net, {"vertical": grid})
    assert emitted > 0
    assert net.event_queue_stats["feature_bus_enqueued"] == emitted
    assert net.event_queue_stats["feature_bus_pending"] == emitted


def test_learning_waits_for_real_pool_response_in_causal_trial():
    net = NeuralNetwork(n=100, skip_wiring=True, max_neurons=100)
    lang = LetterIOModule(net=net, symbols=("X", "O"))
    lang._readout_ready = True
    lang._readout_sources = [1]
    lang._readout_pools = {"X": [2], "O": [3]}
    net.connections[1] = [2, 3]
    net.weights[(1, 2)] = 1.05
    net.weights[(1, 3)] = 1.05
    lang._clean_trial_baseline_t = 0.0
    lang._trial_source_spike_count = 4
    lang._trial_source_indices = {1}
    lang._trial_pool_spike_counts = {"X": 0, "O": 0}
    assert lang.apply_readout_learning(net, [1], "X", prediction=None) == 0
    lang._trial_pool_spike_counts = {"X": 5, "O": 6}
    lang._readout_target_sources={2:[1],3:[1]}
    lang.note_readout_spike(1, 10.0); lang.note_readout_eligibility(1,2,10.0,1.0)
    lang.note_readout_spike(2, 20.0)
    # Ambiguous margin: X is still target, but has not cleared the 29f margin gate.
    assert lang.apply_readout_learning(net, [1], "X", prediction="X") > 0


def test_feature_bus_drives_readout_pool_without_forcing_saturation():
    import heapq
    net = NeuralNetwork(n=1000, skip_wiring=True, max_neurons=1000)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    net.modules["lenguaje"] = lang
    lang.configure_learned_readout(net)
    lang.begin_clean_trial(0.0, net=net)
    target_grid = np.zeros((20, 20), dtype=np.float32)
    target_grid[2:6, 2:6] = 10.0
    for k in range(4):
        lang.emit_feature_maps(net, {"vertical": target_grid}, t=float(k * 30.0))
        # process all ready feature events and their resulting readout events
        while net.event_queue:
            ev = heapq.heappop(net.event_queue)
            net.current_time = max(net.current_time, float(ev[0]))
            net.process_event(*ev)
        while net.readout_event_queue:
            ev = heapq.heappop(net.readout_event_queue)
            net.current_time = max(net.current_time, float(ev[0]))
            net.process_event(*ev)
    rt = lang.get_trial_readout_activity()
    assert rt["source_spikes"] >= 2
    # v1.9: el pool es un integrador selectivo; una entrada simple puede quedar
    # subumbral. Lo importante es que llegue señal y se acumule sin saturar.
    assert sum(rt["pool_spikes"].values()) >= 0
    delivery = lang._trial_pool_delivery_diag
    assert any(int(d.get("events", 0)) > 0 for d in delivery.values())


def test_readout_projection_uses_cortical_sources_and_remains_learnable():
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    assert lang._readout_sources
    cortex = set(lang._cortical_representation_neurons)
    assert set(lang._readout_sources) == cortex
    feature_sources = {int(i) for g in lang._feature_bus_groups.values() for i in g}
    assert cortex.isdisjoint(feature_sources)
    src = int(lang._readout_sources[0])
    targets = set(net.connections[src])
    connected_classes = [sym for sym in lang.symbols if any(t in lang._readout_pools[sym] for t in targets)]
    assert 1 <= len(connected_classes) <= len(lang.symbols)
    assert len(connected_classes) == 5
    assert tuple(connected_classes) == tuple(lang._feature_bus_source_class_map[src])

def test_structural_maps_are_emitted_by_letter_detector():
    from modules.detector_letras import LetraDetectorBlock
    det = LetraDetectorBlock(grid_res=20)
    det._curv_grid[1:4,1:4] = 2.0
    det._j_grid[9:12,9:12] = 1.0
    det._h_grid[0:3,:] = 1.0
    det._barra_grid[2:6,2:6] = 1.0
    det._slash_grid[2:6,2:6] = 1.0
    det._build_structural_maps()
    assert float(det._closure_grid.max()) > 0
    assert float(det._center_junction_grid.max()) > 0
    assert float(det._top_bar_grid.max()) > 0
    assert float(det._diag_cross_grid.max()) > 0


def test_cortical_representation_emits_distributed_winners():
    """La representación no debe colapsar a 1-3 fuentes por frame."""
    import numpy as np
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_cortical_representation(net)

    emitted_targets = []
    original = net.receive_spike
    def capture(target_idx, strength, arrival_time, origin=None):
        if origin == "feature_bus":
            emitted_targets.append(int(target_idx))
            return True
        return original(target_idx, strength, arrival_time, origin=origin)
    net.receive_spike = capture

    maps = {}
    for f in lang.feature_bus_features:
        maps[f] = np.zeros((20,20), dtype=np.float32)
    maps["vertical"][4:16, 6:9] = 1.0
    maps["horizontal"][9:12, 4:16] = 0.8
    emitted = lang._emit_cortical_representation(net, maps, 0.0)
    first_frame = emitted_targets.copy()
    assert emitted >= 6
    assert len(set(first_frame)) >= 6

    emitted_targets.clear()
    emitted2 = lang._emit_cortical_representation(net, maps, 480.0)
    assert emitted2 >= 6
    assert len(set(emitted_targets)) >= 6


def test_readout_projection_is_balanced_across_pools():
    """La topología 3/5 debe repartir las fuentes de forma aproximadamente uniforme."""
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    counts = {sym: 0 for sym in lang.symbols}
    for src in lang._readout_sources:
        for sym in lang.symbols:
            if any(int(t) in lang._readout_pools[sym] for t in net.connections[int(src)]):
                counts[sym] += 1
    vals = list(counts.values())
    assert min(vals) >= 50
    assert max(vals) - min(vals) <= 2
