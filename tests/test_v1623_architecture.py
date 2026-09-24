from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def test_temporal_core_event_order_and_epoch():
    from core.temporal import TemporalCore
    tc=TemporalCore(); tc.begin_trial("t1",100.0); f=tc.begin_frame(0,133.333); a=tc.next_event(120.0,"sensory"); b=tc.next_event(120.0,"synaptic")
    assert f.epoch==1 and f.frame_index==0
    assert b>a and tc.snapshot()["event_seq"]==2

def test_v1623_sparse_stable_fringe_and_persistence():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net=NeuralNetwork(n=1500,skip_wiring=True,max_neurons=1500)
    lang=LetterIOModule(net=net,symbols=("X","O","T","A","E")); lang.configure_learned_readout(net)
    nnz=np.sum(np.abs(lang._cortical_projection)>1e-9,axis=1)
    assert set(nnz.tolist())=={28}
    assert lang.get_readout_topology_audit(net)["projection_version"]>=8
    lang._cortical_core_ema[:10]=0.5; lang._cortical_usage_ema[:10]=0.2
    topo=lang.get_state()["readout_topology"]
    lang2=LetterIOModule(net=net,symbols=("X","O","T","A","E")); lang2.restore_state({"readout_topology":topo},max_idx=1500)
    assert lang2._readout_ready
    assert np.allclose(lang2._cortical_projection,lang._cortical_projection)
    assert np.allclose(lang2._cortical_core_ema,lang._cortical_core_ema)

def test_v1623_static_pattern_keeps_winners_stable_within_trial():
    from core.network import NeuralNetwork
    from modules.language_io import LetterIOModule
    net=NeuralNetwork(n=1600,skip_wiring=True,max_neurons=1600)
    lang=LetterIOModule(net=net,symbols=("X","O","T","A","E"))
    lang.configure_feature_bus(net)
    lang.configure_learned_readout(net)
    maps={f:np.zeros((20,20),dtype=np.float32) for f in lang.feature_bus_features}
    maps["vertical"][3:17,4:6]=8; maps["vertical"][3:17,14:16]=8
    maps["horizontal"][3:5,5:15]=8; maps["horizontal"][9:11,6:14]=8
    maps["corner"][3:17,4:16]=3
    lang.begin_clean_trial(1000.0,net=net)
    sets=[]
    for frame in range(6):
        lang._trial_cortical_counts[:] = 0
        lang._emit_cortical_representation(net,maps,1000.0+frame*33.333)
        sets.append(set(np.flatnonzero(lang._trial_cortical_counts).tolist()))
    def j(a,b):
        return len(a&b)/len(a|b) if a|b else 1.0
    assert all(j(sets[0],s) >= 0.95 for s in sets[1:])
