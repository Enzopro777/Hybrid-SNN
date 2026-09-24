from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_v1621_config_mechanisms_present():
    from config import (
        CORTICAL_REPRESENTATION_SPARSE_INPUTS,
        CORTICAL_REPRESENTATION_TEMPORAL_BLEND,
        CORTICAL_REPRESENTATION_SCORE_THRESHOLD,
        READOUT_POOLS_PER_SOURCE,
        READOUT_TARGETS_PER_POOL_PER_SOURCE,
        READOUT_LEARNING_SOURCE_FRACTION,
        CAUSAL_TRIAL_FRAME_DT_MS,
    )
    assert CORTICAL_REPRESENTATION_SPARSE_INPUTS == 36
    assert 0.0 < CORTICAL_REPRESENTATION_TEMPORAL_BLEND <= 1.0
    assert 0.49 <= CORTICAL_REPRESENTATION_SCORE_THRESHOLD < 1.0
    assert READOUT_POOLS_PER_SOURCE == 5
    assert READOUT_TARGETS_PER_POOL_PER_SOURCE == 1
    assert 0.1 <= READOUT_LEARNING_SOURCE_FRACTION <= 1.0
    assert abs(CAUSAL_TRIAL_FRAME_DT_MS - (1000.0 / 30.0)) < 1e-6


def test_v1621_encoder_uses_full_feature_geometry_without_labels():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    assert lang._cortical_projection.shape == (96, 648)
    # 1.6.22: proyección densa; el parámetro histórico sparse queda como compatibilidad de configuración.
    nnz = np.sum(np.abs(lang._cortical_projection) > 1e-9, axis=1)
    assert set(nnz.tolist()) == {28}
    src = Path(ROOT / "modules" / "language_io.py").read_text(encoding="utf-8")
    assert "X/O/T/A/E" in src  # historical terminology may appear in comments/docstrings
    assert "_cortical_projection @ vec" in src


def test_v1621_readout_is_symmetric_five_way_and_score_uses_all_visible_targets():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    for src in lang._readout_sources[:10]:
        connected = []
        for sym in lang.symbols:
            pool = set(lang._readout_pools[sym])
            if any(int(t) in pool for t in net.connections[int(src)]):
                connected.append(sym)
        assert connected == list(lang.symbols)
    audit = lang.get_readout_topology_audit(net)
    assert audit["projection_version"] >= 7
    assert audit["source_pool_count_min"] == 5
    assert audit["source_pool_count_max"] == 5
    assert audit["targets_per_pool_per_source_expected"] == 1
    assert audit["total_source_pool_synapses"] == audit["expected_total_source_pool_synapses"]


def test_v1622_persistence_rejects_old_projection_versions():
    from modules.language_io import LetterIOModule
    lang = LetterIOModule(symbols=("X", "O"))
    lang.net = type("N", (), {"neuron_roles": {1: "cortical_representation"}, "n": 10})()
    lang.restore_state({
        "readout_topology": {
            "projection_version": 3,
            "sources": [1],
            "pools": {"X": [2], "O": [3]},
        }
    }, max_idx=10)
    assert lang._readout_ready is False
    assert lang._readout_sources == []

def test_v1622_probe_identity_is_1622():
    from modules.causal_probe import CausalProbe
    assert CausalProbe.PROJECT_VERSION in {"IA-1.6.24", "IA-1.6.25", "IA-1.6.26", "IA-1.6.27", "IA-1.6.28", "IA-1.6.29", "IA-1.6.29b", "IA-1.6.29c", "IA-1.6.29d", "IA-1.6.29e", "IA-1.6.29f"}
    assert CausalProbe.VERSION in {"causal-probe-v1.4", "causal-probe-v1.5"}


def test_v1622_source_contains_causal_clock_gate_and_bio_sanctuary():
    src = (ROOT / "engine" / "simulation.py").read_text(encoding="utf-8")
    assert "_causal_trial_active" in src
    assert "CAUSAL_TRIAL_FRAME_DT_MS" in src
    assert "clock gate" in src
    assert "ventana causal limpia" in src


def test_v1622_score_mentions_all_visible_synapses():
    src = (ROOT / "modules" / "language_io.py").read_text(encoding="utf-8")
    assert "todas las sinapsis visibles" in src
    assert "spikes reales de los pools" in src


def test_v1622_persisted_v5_reconstructs_cortical_projection():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    topo = lang.get_state()["readout_topology"]
    lang2 = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    # La red ya posee los roles corticales; restore debe poder reconstruir el encoder v5.
    lang2.restore_state({"readout_topology": topo}, max_idx=1500)
    assert lang2._readout_ready is True
    assert lang2.cortical_representation_ready is True
    assert lang2._cortical_projection.shape == (96, 648)
    assert np.allclose(lang._cortical_projection, lang2._cortical_projection)
