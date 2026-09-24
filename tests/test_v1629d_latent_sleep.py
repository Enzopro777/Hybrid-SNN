import numpy as np

from systems.latent_workspace import LatentWorkspace


def _make_ws():
    return LatentWorkspace(
        input_sources=6, temporal_bins=3, hidden_dim=12, steps=3, top_k=6,
        seed=162904, sleep_interval=2, sleep_nrem_passes=1,
        sleep_replay_per_class=1, sleep_replay_weight=0.05,
        sleep_enabled=True, information_flow_audit=True,
    )


def _bins(offset=0):
    return {
        0: [2+offset, 1, 0], 1: [0, 2, 1], 2: [1, 0, 2],
        3: [0, 1, 1], 4: [1, 1, 0], 5: [0, 1, 2],
    }


def test_information_flow_audit_present():
    ws = _make_ws()
    out = ws.run(_bins(), ["X", "O", "T"], steps=3)
    flow = out["information_flow"]
    assert flow["enabled"]
    assert flow["semantics"] == "proxy_not_mutual_information"
    assert "cortical_to_h0" in flow
    assert "h0_to_latent" in flow


def test_sleep_replay_triggers_and_isolated_from_event_queue():
    ws = _make_ws()
    for target in ["X", "O", "T", "X"]:
        ws.run(_bins(), ["X", "O", "T"], steps=2)
        res = ws.learn_from_trial(target, ["X", "O", "T"], apply=True)
    assert ws._sleep_cycles >= 1
    assert res["sleep"]["applied"] is True
    assert res["sleep"]["event_queue_touched"] is False
    assert sum(len(v) for v in ws._episode_memory.values()) >= 2


def test_state_roundtrip_keeps_sleep_memory():
    ws = _make_ws()
    ws.run(_bins(), ["X", "O", "T"], steps=2)
    ws.learn_from_trial("X", ["X", "O", "T"], apply=True)
    state = ws.state_dict()
    restored = _make_ws()
    restored.restore_state(state)
    assert restored.learning_steps == ws.learning_steps
    assert restored._sleep_cycles == ws._sleep_cycles
    assert len(restored._episode_memory.get("X", [])) == len(ws._episode_memory.get("X", []))
