from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_feature_bus_contains_closure_ring():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert '"closure_ring"' in src

def test_readout_has_no_class_coded_affinity():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert 'set(self.feature_bus_features)' in src
    assert 'allowed_symbols = [sym for sym in self.symbols if feature in self._readout_feature_affinity' not in src

def test_learning_uses_source_spike_counts():
    src=(ROOT/"modules"/"language_io.py").read_text(encoding="utf-8")
    assert '_trial_source_spike_counts' in src
    assert 'np.log1p(count)' in src

def test_detector_has_topological_closure_signals():
    src=(ROOT/"modules"/"detector_letras.py").read_text(encoding="utf-8")
    assert 'closure_ring_density' in src
    assert 'center_void' in src

def test_prediction_does_not_abstain_on_every_nonzero_margin():
    src=(ROOT/"engine"/"simulation.py").read_text(encoding="utf-8")
    assert 'abs(top - second) <= 1e-12 and top <= 0.15' in src


def test_learning_breaks_initial_symmetry_toward_target():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    lang.begin_clean_trial(0.0, net=net)
    active = lang._readout_sources[:8]
    lang._trial_source_indices = set(active)
    lang._trial_source_spike_counts = {int(i): 5 for i in active}
    lang._trial_source_spike_count = 40
    lang._trial_pool_spike_counts = {"X": 5, "O": 6, "T": 2, "A": 2, "E": 2}
    lang._readout_target_sources={int(t): active for t in lang._readout_pools["X"]}
    for src in active[:2]:
        lang.note_readout_spike(src,10.0); lang.note_readout_eligibility(src,lang._readout_pools["X"][0],10.0,1.0)
    lang.note_readout_spike(lang._readout_pools["X"][0],20.0)
    changed = lang.apply_readout_learning(net, active, "X", prediction=None)
    assert changed > 0
    scores = lang.readout_scores(net)
    assert scores["X"] >= max(scores[s] for s in lang.symbols if s != "X")
