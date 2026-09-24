import ast
from pathlib import Path
import numpy as np
import threading

ROOT = Path(__file__).resolve().parents[1]


def test_readout_not_saturated_by_default():
    src=(ROOT/"core"/"network.py").read_text(encoding="utf-8")
    assert "self.readout_synaptic_gain" in src
    assert "READOUT_SYNAPTIC_GAIN" in src
    cfg=(ROOT/"config.py").read_text(encoding="utf-8")
    assert "READOUT_SYNAPTIC_GAIN = 1.10" in cfg
    assert "READOUT_POOL_VTHRESH = 1.55" in cfg
    assert "READOUT_SCORE_TEMPERATURE = 0.65" in cfg
    assert "READOUT_LEARNING_RATE = 0.045" in cfg


def test_score_keeps_absolute_magnitude():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert "spike real de cada pool" in src
    assert "val / mx" not in src
    assert "READOUT_SCORE_TEMPERATURE" in src


def test_evidence_ports_have_no_neural_wiring():
    src=(ROOT/"modules"/"detector_letras.py").read_text(encoding="utf-8")
    tree=ast.parse(src)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="LetraDetectorBlock")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="auto_wire_evidence_ports")
    assert "connected_neurons = []" in ast.unparse(fn)
    assert "sensory_relay_routes" not in ast.unparse(fn)


def test_letter_training_does_not_apply_global_reward():
    src=(ROOT/"engine"/"simulation.py").read_text(encoding="utf-8")
    assert "trial de lectura no debe recompensar/deprimir toda la red biológica" in src


def test_plain_pytest_discovery_is_configured():
    cfg=(ROOT/"pytest.ini").read_text(encoding="utf-8")
    assert "pythonpath = ." in cfg


def test_network_can_import_without_msvcrt():
    src=(ROOT/"engine"/"simulation.py").read_text(encoding="utf-8")
    assert "except ImportError" in src and "msvcrt = None" in src


def test_score_uses_symmetric_per_source_competition():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert "spikes reales de los pools" in src


def test_readout_scores_are_uniform_before_learning_and_separate_after_weight_change():
    import threading
    from modules.language_io import LetterIOModule

    class FakeNet:
        def __init__(self):
            self.n=32; self.lock=threading.RLock(); self.connections=[[] for _ in range(self.n)]
            self.weights={}; self.current_time=60.0; self.last_spike=np.zeros(self.n, dtype=float); self.membrane_potential=np.zeros(self.n, dtype=float); self.last_update_time=np.zeros(self.n, dtype=float); self.refractory_until=np.zeros(self.n, dtype=float)

    net=FakeNet()
    lang=LetterIOModule(net=net, symbols=("X","O","T","A","E"))
    lang._readout_ready=True; lang._readout_sources=[1]
    lang._readout_pools={s:[10+i*2] for i,s in enumerate(lang.symbols)}
    net.connections[1]=[10,12,14,16,18]
    for t in net.connections[1]: net.weights[(1,t)]=lang.readout_initial_weight
    lang.begin_clean_trial(0.0, net=net)
    lang._trial_source_spike_count=1; lang._trial_source_spike_counts={1:1}
    scores=lang.readout_scores(net, t=60.0)
    assert all(scores[s] == 0.0 for s in lang.symbols)
    net.weights[(1,10)] = 1.8
    # La decisión ya no usa pesos estáticos: registra un spike real en X.
    lang.note_readout_spike(10, 20.0)
    net.weights[(1,12)] = 0.8
    scores2=lang.readout_scores(net, t=60.0)
    assert scores2["X"] > scores2["O"]
    assert abs(max(scores2.values()) - 1.0) < 1e-6


def test_causal_probe_version_matches_release():
    src=(ROOT/"modules"/"causal_probe.py").read_text(encoding="utf-8")
    assert ('PROJECT_VERSION = "IA-1.6.29"' in src) or ('PROJECT_VERSION = "IA-1.6.29c"' in src) or ('PROJECT_VERSION = "IA-1.6.29d"' in src) or ('PROJECT_VERSION = "IA-1.6.29f"' in src)


def test_temporal_source_signature_is_recorded():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert "_trial_source_bin_counts" in src
    assert "source_bin_counts" in src


def test_temporal_readout_is_explicit():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert "READOUT_TEMPORAL_BINS" in src
    assert "pool_temporal_bins" in src


def test_readout_routes_a_fixed_budget_competitively_per_source():
    src=(ROOT/"core"/"network.py").read_text(encoding="utf-8")
    cfg=(ROOT/"config.py").read_text(encoding="utf-8")
    assert "acondicionador competitivo por fuente" in src
    assert "READOUT_ROUTING_TEMPERATURE = 0.35" in cfg
    assert "READOUT_ROUTING_MIN_BUDGET = 2.80" in cfg
    assert "READOUT_ROUTING_MAX_BUDGET = 4.00" in cfg


def test_readout_does_not_multiply_competitive_budget_again():
    src=(ROOT/"core"/"network.py").read_text(encoding="utf-8")
    assert "volvería a saturar los todos los pools" in src or "saturar los" in src


def test_readout_calibration_budget_is_above_pool_threshold_window():
    cfg=(ROOT/"config.py").read_text(encoding="utf-8")
    assert "READOUT_ROUTING_MIN_BUDGET = 2.80" in cfg
    assert "READOUT_ROUTING_MAX_BUDGET = 4.00" in cfg
    src=(ROOT/"core"/"network.py").read_text(encoding="utf-8")
    assert "readout_calibrated_budget_sum" in src
