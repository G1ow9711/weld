"""Compile original one-parent witness alternatives without copying output count.

This compiler is explicitly restricted to the first registered TRAIN21 trial.
Multi-parent training must provide complete compatible parent witnesses first.
"""
import numpy as np
from weld_pipeline.sjg_structured_contact_1057 import TRANSITIONS

def compile_one_parent(cells,family,sequences,y,original_parent_count,original_child_count):
    assert original_parent_count==original_child_count==1,'Do not reuse single-parent compiler for broader pilot'
    n=len(cells);node=np.zeros((n,3,4),np.float32);terminal=np.zeros((len(sequences),4),np.float32)
    occ=np.zeros((n,3),np.float32);positions=np.tile([1/3,2/3],(n,1));pm=np.zeros((n,2),bool)
    witnesses=[];boundary_events=0;internal_events=0;maxsnap=0.
    for k,s in enumerate(sequences):
        a,b=s['start'],s['end'];events=[]
        for i,j in np.argwhere(y['position_mask'][a:b]):
            i+=a;q=cells[i,0]+y['positions'][i,j]*(cells[i,1]-cells[i,0]);events.append((q,y['event'][i,j].copy(),i,j))
        if not events:continue
        assert not s['cyclic'] and len(events)==2,'This real trial is one open straight child'
        witnesses.append(k);coords=np.r_[cells[a:b,0],cells[b-1,1]]
        for q,bits,i,j in events:
            idx=int(np.argmin(abs(coords-q)));err=abs(coords[idx]-q)
            if err<=1e-6:
                boundary_events+=1;maxsnap=max(maxsnap,err)
                if idx==len(coords)-1:terminal[k]=np.maximum(terminal[k],bits)
                else:node[a+idx,0]=np.maximum(node[a+idx,0],bits)
            else:
                internal_events+=1;node[i,j+1]=bits;positions[i,j]=y['positions'][i,j];pm[i,j]=True
        lo=min(q for q,_,_,_ in events);hi=max(q for q,_,_,_ in events)
        cuts=np.c_[cells[a:b,0],cells[a:b,0,None]+(cells[a:b,1]-cells[a:b,0])[:,None]*positions[a:b],cells[a:b,1]]
        mid=(cuts[:,:-1]+cuts[:,1:])*.5;occ[a:b]=((mid>lo)&(mid<hi)).astype(np.float32)
        # Every target must be an actual marked path in the inference state graph.
        flat=occ[a:b].reshape(-1).astype(int);ev=np.r_[node[a:b].reshape(-1,4),terminal[k][None]];prev=0
        for j,state in enumerate(np.r_[flat,0]):
            bits=tuple(int(v) for v in ev[j]);assert (prev,int(state),bits) in TRANSITIONS,(k,j,prev,state,bits)
            prev=int(state)
    return dict(occupation=occ,node_event=node,terminal_event=terminal,positions=positions,position_mask=pm),dict(witness_sequences=witnesses,boundary_events=boundary_events,internal_events=internal_events,maximum_numerical_snap_mm=maxsnap,anchors_added_from_GT=0,original_output_multiplicity=1)


def select_output_responsibility(parent_witnesses,costs):
    """Tiny exact injective witness selection for multiplicity controls.

    A witness is a complete path key, not a family ID: distinct paths/parents may
    occupy one family. This is an offline target-assignment control, not inference.
    """
    import itertools
    candidates=[]
    for assignment in itertools.product(*parent_witnesses):
        if len(set(assignment))!=len(assignment):continue
        candidates.append((sum(costs[i,w] for i,w in enumerate(assignment)),assignment))
    if not candidates:raise ValueError('No representation for required parent multiplicity')
    return min(candidates,key=lambda x:(x[0],x[1]))[1]

def controls():
    assert len(select_output_responsibility([list(range(17))],{(0,j):float(j) for j in range(17)}))==1
    a=select_output_responsibility([[0,1],[0,1]],{(i,j):0. for i in range(2) for j in range(2)})
    assert len(set(a))==2
    return dict(one_parent_17_alternatives_one_responsibility=True,two_same_geometry_distinct_side_path_keys_two_responsibilities=True,no_material_identity_fabricated=True)
