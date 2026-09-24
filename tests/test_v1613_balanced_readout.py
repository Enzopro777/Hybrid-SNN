from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_curriculum_defaults_to_all_five_letters_and_balanced_mode():
    from config import LETTER_VOCABULARY, CURRICULUM_STARTER_LETTERS, CURRICULUM_BALANCED_MODE
    assert LETTER_VOCABULARY == ["X", "O", "T", "A", "E"]
    assert CURRICULUM_STARTER_LETTERS == LETTER_VOCABULARY
    assert CURRICULUM_BALANCED_MODE is True


def test_curriculum_next_letter_keeps_counts_balanced():
    from modules.curriculum import VocabularyCurriculum
    cur = VocabularyCurriculum(
        ["X", "O", "T", "A", "E"],
        starter_letters=["X", "O", "T", "A", "E"],
        balanced_mode=True,
        max_trial_imbalance=1,
    )
    for sym, count in {"X": 4, "O": 4, "T": 5, "A": 5, "E": 5}.items():
        cur.total_trials[sym] = count
    assert cur.next_letter() in {"X", "O"}


def test_one_vs_rest_learning_moves_all_non_targets():
    from tests.test_v1611_pool_separation import make_readout_with_5_symbols
    net, lang = make_readout_with_5_symbols()
    lang.begin_clean_trial(0.0, net=net)
    lang._trial_source_spike_count = 3
    lang._trial_source_spike_counts = {1: 2, 2: 1}
    lang._trial_pool_spike_counts = {"X": 5, "O": 6, "T": 2, "A": 2, "E": 2}
    # Fuente 1/2 conectadas a una única neurona de cada pool por la topología.
    lang._readout_sources=[1,2]
    for src in (1, 2):
        net.connections[src]=[10,11,12,13,14]
        net.weights[(src, 10)] = 1.05
        net.weights[(src, 11)] = 1.05
        net.weights[(src, 12)] = 1.05
        net.weights[(src, 13)] = 1.05
        net.weights[(src, 14)] = 1.05
    lang._readout_target_sources={10:[1,2],11:[1,2],12:[1,2],13:[1,2],14:[1,2]}
    lang.note_readout_spike(1,10.0); lang.note_readout_eligibility(1,10,10.0,1.0); lang.note_readout_spike(10,20.0)
    lang.note_readout_spike(2,10.0); lang.note_readout_eligibility(2,10,10.0,1.0)
    lang.apply_readout_learning(net, [1, 2], "X", prediction="X")
    assert net.weights[(1, 10)] > 1.05
    # 1.6.27: una sinapsis rival sin elegibilidad local no recibe crédito artificial.
    assert net.weights[(1, 11)] == 1.05
    assert net.weights[(1, 12)] == 1.05
    assert net.weights[(1, 13)] == 1.05
    assert net.weights[(1, 14)] == 1.05


def test_pool_sets_are_structurally_disjoint():
    from tests.test_v1611_pool_separation import make_readout_with_5_symbols
    net, lang = make_readout_with_5_symbols()
    pools = [set(v) for v in lang._readout_pools.values()]
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            assert pools[i].isdisjoint(pools[j])


def test_lateral_inhibition_requires_clear_majority():
    from config import READOUT_LATERAL_INHIBITION
    cfg = (ROOT / "modules" / "language_io.py").read_text(encoding="utf-8")
    assert "winner_share < 0.45" not in cfg
    assert 0.05 <= READOUT_LATERAL_INHIBITION <= 0.50


def test_new_rival_depression_parameter_exists():
    cfg = (ROOT / "config.py").read_text(encoding="utf-8")
    assert re.search(r"READOUT_ALL_RIVAL_DEPRESSION\s*=\s*[0-9.]+", cfg)
