from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_cortical_representation_is_initialized_and_agnostic():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    assert lang.cortical_representation_ready
    assert len(lang._cortical_representation_neurons) == 96
    assert set(lang._readout_sources) == set(lang._cortical_representation_neurons)
    assert lang._cortical_projection.shape == (96, 648)


def test_feature_bus_neurons_are_not_direct_readout_sources():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net = NeuralNetwork(n=1500, skip_wiring=True, max_neurons=1500)
    lang = LetterIOModule(net=net, symbols=("X", "O", "T", "A", "E"))
    lang.configure_learned_readout(net)
    bus = {int(i) for g in lang._feature_bus_groups.values() for i in g}
    assert bus.isdisjoint(set(lang._readout_sources))


def test_cortical_projection_has_no_symbol_labels_in_state():
    from modules.language_io import LetterIOModule
    lang = LetterIOModule(symbols=("X", "O", "T", "A", "E"))
    assert not hasattr(lang, "_cortical_class_affinity")
    assert not hasattr(lang, "_cortical_symbol_templates")
