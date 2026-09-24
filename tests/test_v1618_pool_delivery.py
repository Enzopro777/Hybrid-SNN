from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_language_io_exposes_pool_delivery_diagnostics():
    from modules.language_io import LetterIOModule
    lang = LetterIOModule(symbols=("X", "O"))
    lang._readout_pools = {"X": [10], "O": [11]}
    lang._readout_ready = True
    lang._clean_trial_baseline_t = 0.0
    lang._trial_pool_delivery_diag = {s: lang._new_pool_delivery_diag() for s in lang.symbols}
    lang._trial_pool_delivery_samples = []
    class N: pass
    lang.net = N()
    lang.note_readout_delivery(10, 1.0, 0.5, 0.2, 0.18, 0.68, 1.55, False, False)
    d = lang.get_trial_readout_activity()["pool_delivery_diagnostics"]["X"]
    assert d["events"] == 1
    assert d["strength_sum"] == 0.5
    assert d["fired_events"] == 0
    assert d["near_threshold_events"] == 0


def test_pool_delivery_sample_carries_membrane_threshold_state():
    from modules.language_io import LetterIOModule
    lang = LetterIOModule(symbols=("X",))
    lang._readout_pools = {"X": [10]}
    lang._readout_pool_neuron_fatigue = {10: 0.2}
    lang._pool_inhibition_state = {"X": 0.3}
    lang._clean_trial_baseline_t = 0.0
    lang._trial_pool_delivery_diag = {"X": lang._new_pool_delivery_diag()}
    lang._trial_pool_delivery_samples = []
    lang.note_readout_delivery(10, 3.0, 1.2, 0.1, 0.09, 1.29, 1.55, False, False)
    sample = lang.get_trial_readout_activity()["pool_delivery_samples"][0]
    assert sample["pre_v"] == 0.1
    assert sample["post_v"] == 1.29
    assert sample["threshold"] == 1.55
    assert sample["pool_inhibition"] == 0.3
    assert sample["neuron_fatigue"] == 0.2
