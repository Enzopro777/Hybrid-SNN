from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def test_v1624_identity_and_clean_defaults():
    from config import CLEAN_EXPERIMENT_DEFAULT, CLEAN_EXPERIMENT_REMOVE_CALIBRATION
    from modules.causal_probe import CausalProbe
    assert CausalProbe.PROJECT_VERSION in {"IA-1.6.24", "IA-1.6.25", "IA-1.6.26", "IA-1.6.27", "IA-1.6.28", "IA-1.6.29", "IA-1.6.29b", "IA-1.6.29c", "IA-1.6.29d", "IA-1.6.29e", "IA-1.6.29f"}
    assert CLEAN_EXPERIMENT_DEFAULT is True
    assert CLEAN_EXPERIMENT_REMOVE_CALIBRATION is True

def test_v1624_activity_floor_and_stable_anchor():
    from config import CORTICAL_CORE_SIZE, CORTICAL_MIN_EMIT, CORTICAL_REPRESENTATION_TOP_K
    assert CORTICAL_CORE_SIZE == 6
    assert CORTICAL_MIN_EMIT == 6
    assert CORTICAL_REPRESENTATION_TOP_K >= 8

def test_v1624_static_pattern_has_active_cortical_emission():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net=NeuralNetwork(n=1600,skip_wiring=True,max_neurons=1600)
    lang=LetterIOModule(net=net,symbols=("X","O","T","A","E"))
    lang.configure_learned_readout(net)
    maps={f:np.zeros((20,20),dtype=np.float32) for f in lang.feature_bus_features}
    maps["vertical"][3:17,4:6]=8; maps["vertical"][3:17,14:16]=8
    maps["horizontal"][3:5,5:15]=8; maps["horizontal"][9:11,6:14]=8
    maps["corner"][3:17,4:16]=3
    lang.begin_clean_trial(1000.0,net=net)
    sets=[]
    for frame in range(6):
        lang._emit_cortical_representation(net,maps,1000.0+frame*33.333)
        sets.append(set(np.flatnonzero(lang._trial_cortical_counts).tolist()))
    assert all(len(s) >= 6 for s in sets)
    def j(a,b): return len(a&b)/len(a|b) if a|b else 1.0
    assert all(j(sets[0],s) >= 0.75 for s in sets[1:])

def test_v1624_readout_projection_version_and_persistence_contract():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net=NeuralNetwork(n=1500,skip_wiring=True,max_neurons=1500)
    lang=LetterIOModule(net=net,symbols=("X","O","T","A","E")); lang.configure_learned_readout(net)
    audit=lang.get_readout_topology_audit(net)
    assert audit["projection_version"] >= 7
    topo=lang.get_state()["readout_topology"]
    assert topo["encoder_version"].startswith(("1.6.25-", "1.6.26-", "1.6.27-"))
    assert topo["core_size"] == 6
