import numpy as np
import threading

from modules.language_io import LetterIOModule
from modules.curriculum import VocabularyCurriculum


class _FakeNet:
    def __init__(self):
        self.n = 32
        self.lock = threading.RLock()
        self.connections = [[] for _ in range(self.n)]
        self.weights = {}


def _tiny_net():
    return _FakeNet()


def test_curriculum_next_letter_remains_balanced():
    c = VocabularyCurriculum(["X", "O"], starter_letters=["X", "O"], window=8)
    for _ in range(12):
        c.record_trial("O", 1.0, True)
    # X is under-sampled and should receive a larger weight than O.
    # We cannot guarantee a single random draw, so inspect weights indirectly
    # by monkey-patching random.choices.
    import modules.curriculum as mod
    chosen = []
    old = mod.random.choices
    try:
        mod.random.choices = lambda seq, weights, k=1: (chosen.append((list(seq), list(weights))) or [seq[int(np.argmax(weights))]])
        out = c.next_letter()
    finally:
        mod.random.choices = old
    assert out == "X"
    assert chosen[0][1][0] > chosen[0][1][1]


def test_readout_learning_uses_active_targets_only_and_has_contrastive_competitor():
    net = _tiny_net()
    lang = LetterIOModule(net=net, symbols=("X", "O"))
    # Build a deterministic mini readout manually; avoid needing a full spatial
    # setup because this test targets learning semantics only.
    lang._readout_ready = True
    lang._readout_sources = [1, 2]
    lang._readout_pools = {"X": [10, 11], "O": [20, 21]}
    lang._clean_trial_baseline_t = 0.0
    lang._trial_source_spike_counts = {1: 5, 2: 3}
    lang._trial_source_spike_count = 8
    lang._trial_pool_spike_counts = {"X": 4, "O": 6}
    lang._trial_pool_indices = {"X": {10}, "O": {20, 21}}
    for src in lang._readout_sources:
        net.connections[src] = [10, 11, 20, 21]
        for tgt in net.connections[src]:
            net.weights[(src, tgt)] = lang.readout_initial_weight

    lang._readout_target_sources={t:[1,2] for t in [10,11,20,21]}
    lang.note_readout_spike(1,10.0); lang.note_readout_eligibility(1,10,10.0,1.0); lang.note_readout_spike(10,20.0)
    changed = lang.apply_readout_learning(net, [1, 2], "X", prediction="O")
    assert changed > 0
    # Target: active X neuron 10 changes; inactive X neuron 11 does not receive
    # the direct target reinforcement on this trial. Competitor O uses its active
    # neurons.
    assert net.weights[(1, 10)] > lang.readout_initial_weight
    assert net.weights[(1, 11)] == lang.readout_initial_weight
    assert net.weights[(1, 20)] == lang.readout_initial_weight
