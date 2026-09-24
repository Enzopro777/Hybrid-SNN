from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _maps(kind):
    base = np.zeros((20,20), dtype=np.float32)
    if kind == "X":
        for i in range(20):
            base[i,i] = 1.0
            base[i,19-i] = 1.0
    elif kind == "E":
        base[:,2:4] = 1.0
        base[2:5,:14] = 1.0
        base[9:12,:12] = 1.0
        base[16:19,:14] = 1.0
    else:
        base[9:12,2:18] = 1.0
    feats = (
        "vertical","horizontal","diag_back","diag_slash","curve","corner","junction","occupancy",
        "center_junction","closure","top_bar","mid_bar","bottom_bar","center_vertical",
        "left_vertical","diag_cross","apex","closure_ring"
    )
    out={k:np.zeros_like(base) for k in feats}
    out["occupancy"] = base
    out["diag_back"] = base if kind=="X" else np.zeros_like(base)
    out["diag_slash"] = base if kind=="X" else np.zeros_like(base)
    out["left_vertical"] = base if kind=="E" else np.zeros_like(base)
    out["vertical"] = base if kind=="E" else np.zeros_like(base)
    out["horizontal"] = base if kind=="E" else np.zeros_like(base)
    out["top_bar"] = base if kind=="E" else np.zeros_like(base)
    out["mid_bar"] = base if kind=="E" else np.zeros_like(base)
    out["bottom_bar"] = base if kind=="E" else np.zeros_like(base)
    out["center_junction"] = base if kind=="X" else np.zeros_like(base)
    out["diag_cross"] = base if kind=="X" else np.zeros_like(base)
    return out


def test_v1622_multiscale_separator_is_agnostic_and_deterministic():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X","O","T","A","E"))
    lang.configure_learned_readout(net)
    assert tuple(lang.feature_bus_bins) == (6,6)
    assert lang._cortical_input_dim == 648
    assert lang._cortical_projection.shape == (96,648)
    assert lang._cortical_separator_projection.shape == (96,648)
    assert lang._cortical_separator_phase.shape == (96,)
    a = lang._build_cortical_input_vector(_maps("X"))
    b = lang._build_cortical_input_vector(_maps("E"))
    assert np.linalg.norm(a-b) > 0.10

    lang2 = LetterIOModule(net=net, symbols=("X","O","T","A","E"))
    lang2._cortical_representation_neurons = list(lang._cortical_representation_neurons)
    lang2._cortical_input_dim = lang._cortical_input_dim
    lang2._initialize_cortical_projection(96)
    assert np.allclose(lang._cortical_projection, lang2._cortical_projection)
    assert np.allclose(lang._cortical_separator_projection, lang2._cortical_separator_projection)


def test_v1622_detector_x_audit_exists_without_readout_coupling():
    from modules.detector_letras import LetraDetectorBlock
    d = LetraDetectorBlock()
    assert "x_signal_audit" in d.health
    src=(ROOT/"modules"/"detector_letras.py").read_text(encoding="utf-8")
    assert "No alimenta el readout" in src
    assert "X-audit=" in src


def test_v1622_probe_identity():
    from modules.causal_probe import CausalProbe
    assert CausalProbe.PROJECT_VERSION in {"IA-1.6.29", "IA-1.6.29b", "IA-1.6.29c", "IA-1.6.29d", "IA-1.6.29e", "IA-1.6.29f"}


def test_v1622_separator_keeps_static_input_active_across_frames():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X","O","T","A","E"))
    lang.configure_cortical_representation(net)
    maps = _maps("X")
    counts=[]
    for t in (0.0, 33.333, 66.666):
        counts.append(lang._emit_cortical_representation(net, maps, t))
    assert all(c >= 1 for c in counts)


def test_v1622_separator_stats_are_persisted():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X","O","T","A","E"))
    lang.configure_learned_readout(net)
    lang._cortical_separator_initialized = True
    lang._cortical_separator_mean[:] = 0.123
    lang._cortical_separator_var[:] = 0.456
    topo = lang.get_state()["readout_topology"]
    lang2 = LetterIOModule(net=net, symbols=("X","O","T","A","E"))
    lang2.restore_state({"readout_topology": topo}, max_idx=1500)
    assert lang2._readout_ready
    assert np.allclose(lang2._cortical_separator_mean, 0.123)
    assert np.allclose(lang2._cortical_separator_var, 0.456)
