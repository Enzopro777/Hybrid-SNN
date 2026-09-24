from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_probe_identity_is_1620():
    from modules.causal_probe import CausalProbe
    assert CausalProbe.PROJECT_VERSION in {"IA-1.6.29", "IA-1.6.29b", "IA-1.6.29c", "IA-1.6.29d", "IA-1.6.29e", "IA-1.6.29f"}
    assert CausalProbe.VERSION in {"causal-probe-v1.4", "causal-probe-v1.5"}


def test_runtime_contract_mentions_cortical_source_selection():
    from modules.causal_probe import CausalProbe
    c = CausalProbe()
    contract = c._runtime_contract(None)
    assert contract["project_version"] in {"IA-1.6.29", "IA-1.6.29b", "IA-1.6.29c", "IA-1.6.29d", "IA-1.6.29e", "IA-1.6.29f"}
    assert contract["readout_contract"]["source_selection"] == "cortical_representation"


def test_topology_audit_exposes_roles_and_counts():
    from modules.language_io import LetterIOModule
    class Net:
        n = 20
        neuron_roles = {1: "cortical_representation", 2: "cortical_representation", 3: "monitor_immortal"}
        connections = [[] for _ in range(20)]
    lang = LetterIOModule(symbols=("X", "O"))
    lang.net = Net()
    lang._readout_sources = [1, 2, 3]
    lang._readout_pools = {"X": [10], "O": [11]}
    lang.symbols = ["X", "O"]
    lang.net.connections[1] = [10, 11]
    lang.net.connections[2] = [10]
    lang.net.connections[3] = [11]
    audit = lang.get_readout_topology_audit()
    assert audit["source_role_counts"]["monitor_immortal"] == 1
    assert audit["excluded_source_count"] == 1
    assert audit["pool_sizes"] == {"X": 1, "O": 1}


def test_trial_snapshot_carries_cortical_source_identity():
    from modules.language_io import LetterIOModule
    lang = LetterIOModule(symbols=("X", "O"))
    lang._readout_sources = [1, 2]
    lang._readout_pools = {"X": [10], "O": [11]}
    lang._readout_ready = True
    lang._clean_trial_baseline_t = 0.0
    lang._trial_source_indices = {1, 2}
    lang._trial_source_spike_counts = {1: 3, 2: 2}
    snap = lang.get_trial_readout_activity()
    assert snap["source_indices"] == [1, 2]
    assert snap["source_spike_counts"]["1"] == 3
    assert "readout_topology_audit" in snap
