from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def _setup_lang(n=1500):
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net=NeuralNetwork(n=n,skip_wiring=True,max_neurons=n)
    lang=LetterIOModule(net=net,symbols=("X","O","T","A","E"))
    lang.configure_learned_readout(net)
    return net,lang

def test_v1625_temporal_readout_uses_real_pool_spikes_only():
    net,lang=_setup_lang()
    lang.begin_clean_trial(1000.0,net=net); lang.set_trial_frame_expectation(60)
    # Simula pool O disparando aunque X conserve pesos ligeramente mayores.
    oidx=lang._readout_pools["O"][0]
    for k in range(3):
        lang.note_readout_spike(oidx, 1000.0 + 120.0 + k*33.333)
    scores=lang.readout_scores(net,t=1300.0)
    assert scores["O"] > 0.0
    assert scores["X"] == 0.0
    assert max(scores,key=scores.get)=="O"

def test_v1625_pre_post_eligibility_requires_post_spike():
    net,lang=_setup_lang()
    lang.begin_clean_trial(1000.0,net=net)
    src=lang._readout_sources[0]
    tgt=lang._readout_pools["X"][0]
    lang.note_readout_eligibility(src,tgt,1010.0,1.0)
    assert lang._trial_synapse_eligibility == {}
    lang.note_readout_spike(src,1000.0)
    lang.note_readout_spike(tgt,1020.0)
    assert lang._trial_synapse_eligibility.get((src,tgt),0.0) > 0.0

def test_v1625_identity_prototypes_are_agnostic_and_not_last_trial_anchor():
    from modules.language_io import LetterIOModule
    lang=LetterIOModule(symbols=("X","O"))
    a=np.zeros(96,dtype=np.float32); a[:12]=1
    b=np.zeros(96,dtype=np.float32); b[40:52]=1
    lang.begin_clean_trial(0.0)
    lang._select_identity_prototype(a)
    first=lang._cortical_active_identity
    lang.begin_clean_trial(100.0)
    lang._select_identity_prototype(b)
    second=lang._cortical_active_identity
    assert first != second
    assert len(lang._cortical_identity_prototypes)==2

def test_v1625_slow_synaptic_scaling_preserves_row_order():
    net,lang=_setup_lang()
    src=lang._readout_sources[0]
    pools=[lang._readout_pools[s][0] for s in lang.symbols]
    vals=[0.4,0.7,1.1,1.5,1.8]
    for tgt,v in zip(pools,vals): net.weights[(src,tgt)]=v
    lang._cortical_trials_completed=8
    changed=lang._apply_slow_readout_synaptic_scaling()
    out=[net.weights[(src,t)] for t in pools]
    # 29f intentionally keeps legacy slow scaling out of the experimental path.
    assert changed == 0
    assert out == vals

def test_v1625_state_requires_projection_v8():
    net,lang=_setup_lang()
    topo=lang.get_state()["readout_topology"]
    assert topo["projection_version"]==10
    assert topo["encoder_version"].startswith("1.6.27-")
