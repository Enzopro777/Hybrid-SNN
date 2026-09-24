import ast
from pathlib import Path
import numpy as np

from modules.language_io import LetterIOModule

ROOT = Path(__file__).resolve().parents[1]


def test_feature_bus_handles_short_positions_array():
    class MiniNet:
        n = 10000
        positions = np.zeros((9490, 3), dtype=np.float32)
        active = np.ones(10000, dtype=bool)
        heart_mask = np.zeros(10000, dtype=bool)
        connections = [[] for _ in range(10000)]
        energy = np.ones(10000, dtype=np.float32)
        tau_m = np.ones(10000, dtype=np.float32)
        v_thresh = np.ones(10000, dtype=np.float32) * 20
        refr_period = np.ones(10000, dtype=np.float32)

        def register_noncompetitive_neurons(self, ids, role=None):
            return None

    net = MiniNet()
    # Put enough available neurons in the intended feature-bus z band.
    net.positions[:, 2] = 70.0
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_feature_bus(net)
    assert lang.feature_bus_ready
    assert sum(len(v) for v in lang._feature_bus_groups.values()) == 18 * 36
    assert all(0 <= i < len(net.positions) for v in lang._feature_bus_groups.values() for i in v)


def test_detector_process_does_not_mask_bus_error_with_unbound_logging():
    src = (ROOT / "modules" / "detector_letras.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LetraDetectorBlock")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process")
    local_imports = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Import) and any(alias.name == "logging" for alias in n.names)
    ]
    assert not local_imports


def test_feature_bus_exception_handler_uses_module_logging():
    src = (ROOT / "modules" / "detector_letras.py").read_text(encoding="utf-8")
    assert 'logging.debug(f"[FeatureBus] emisión omitida: {_bus_exc}")' in src
