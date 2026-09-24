def test_queue_backlog_coalescing_reduces_ready_duplicates():
    from core.network import NeuralNetwork
    net = NeuralNetwork(n=20, skip_wiring=True, max_neurons=20)
    with net.lock:
        net.current_time = 10.0
        net.event_queue = [(10.0, 3, 1.0), (10.0, 3, 2.0), (10.0, 4, 1.0), (11.0, 5, 9.0)] * 4000
    import heapq
    heapq.heapify(net.event_queue)
    before = len(net.event_queue)
    removed = net.coalesce_ready_event_backlog(max_events=1000)
    after = len(net.event_queue)
    assert removed > 0
    assert after < before
    assert net.event_queue_stats["ready_events_coalesced"] == removed


def test_queue_stats_include_backlog_relief_metrics():
    from core.network import NeuralNetwork
    net = NeuralNetwork(n=10, skip_wiring=True, max_neurons=10)
    stats = net.get_event_queue_stats()
    assert "ready_events_coalesced" in stats
    assert "backlog_relief_passes" in stats


def test_probe_contract_declares_v11_backlog_policy():
    from modules.causal_probe import CausalProbe
    summary = CausalProbe().summarize()
    contract = summary["runtime_contract"]
    assert contract["readout_contract"]["queue_backpressure"]["ready_backlog_coalescing"] is True
    assert contract["readout_contract"]["queue_backpressure"]["severe_load_watermark"] == 23000
