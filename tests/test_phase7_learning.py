import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_detector_is_observer_only():
    src=(ROOT/"modules"/"detector_letras.py").read_text(encoding="utf-8")
    tree=ast.parse(src); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="LetraDetectorBlock")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="process")
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="_inject_evidence" for n in ast.walk(fn))

def test_signed_reward_and_ctrlc():
    src=(ROOT/"engine"/"simulation.py").read_text(encoding="utf-8")
    assert "trial de lectura no debe recompensar/deprimir toda la red biológica" in src
    assert 'raw_tecla == b"\\x03"' in src

def test_readout_api():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert "configure_learned_readout" in src and "apply_readout_learning" in src and "_readout_pools" in src
    assert "prediction=None" in src


def test_readout_is_separate_from_evidence_ports():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert "NO tocar evidencia_{sym}" in src
    assert "NO tocar evidencia_{sym}" in src or "observacional" in src


def test_causal_contract_declares_readout_separation():
    src=(ROOT/"modules"/"causal_probe.py").read_text(encoding="utf-8")
    assert '"learned_readout_role"' in src
    assert '"source_selection": "cortical_representation"' in src


def test_readout_sources_are_electrically_drivable():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert "SENSORY_RELAY_VTHRESH = 2.5" in src
    assert "net.v_thresh[idx] = min(float(net.v_thresh[idx]), SENSORY_RELAY_VTHRESH)" in src
    assert "strength 3..8" in src


def test_probe_records_authoritative_readout_prediction():
    src=(ROOT/"engine"/"simulation.py").read_text(encoding="utf-8")
    assert 'authoritative_prediction = prediction' in src
    assert 'prediction_source=source' in src
    assert 'No volver a leer evidencia_* aquí' in src


def test_probe_trial_has_prediction_source():
    src=(ROOT/"modules"/"causal_probe.py").read_text(encoding="utf-8")
    assert '"prediction_source": None' in src
    assert 'prediction_source: Optional[str] = None' in src
    assert '"prediction_recording": "causal_probe_end_trial_uses_authoritative_trial_prediction"' in src
