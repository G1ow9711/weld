"""Posthoc endpoint-event upper bound; never a model or deployment output."""
from collections import Counter
import numpy as np
from scipy.signal import find_peaks
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC


def main():
    run=OUT/'run43';inventory=read(OUT/'inventory.json');rows=[]
    assert read(run/'all_predictions_frozen.json')['all_before_GT']
    for si in TRAIN+VALID+ARC:
        manifest=read(run/f'source_{si:02d}_fields.json')['files'];events={}
        profiles=read(OUT/f'run34/source_{si:02d}/profiles.json')
        for item in manifest:
            path=run/item['file'];assert digest(path)==item['sha256']
            with np.load(path) as z:q=z['q_mm'];p=z['probability'];offset=z['offset']
            spacing=float(q[1]-q[0]);distance=max(1,int(np.ceil(2/max(spacing,1e-6))))
            starts=find_peaks(np.r_[-1.,p[:,1],-1.],height=.5,distance=distance)[0]-1
            ends=find_peaks(np.r_[-1.,p[:,2],-1.],height=.5,distance=distance)[0]-1
            lo=np.clip(q[starts]+np.clip(2*offset[starts,0],-4,4),q[0],q[-1])
            hi=np.clip(q[ends]+np.clip(2*offset[ends,1],-4,4),q[0],q[-1])
            profile=profiles[item['profile']];origin=np.asarray(profile['p0']);direction=np.asarray(profile['direction'])
            events[item['profile']]=dict(starts=starts,ends=ends,lo=lo,hi=hi,
                a=origin+lo[:,None]*direction,b=origin+hi[:,None]*direction,
                prefix=np.r_[0.,np.cumsum(p[:,0],dtype=np.float64)])
        # Original GT is consulted only after every model field was frozen.
        target=read(ROOT/inventory[si]['target']['target_path'])
        supervision=read(OUT/f'run35/source_{si:02d}_supervision.json')
        options={(r['parent'],r['child']):r['profiles'] for r in supervision['children']}
        counts=Counter();children=[]
        for pi,parent in enumerate(target['parents']):
            for ci,child in enumerate(parent['children']):
                counts['G']+=1;available_start=available_end=pair=accepted_pair=False
                has_profile=bool(options.get((pi,ci)));has_selected=False
                if child['original_kind']=='line':
                    expected=np.asarray(child['world_representation']['endpoints_mm'])
                    for ni in options.get((pi,ci),[]):
                        if ni not in events:continue
                        has_selected=True;e=events[ni];direction=np.asarray(profiles[ni]['direction'])
                        endpoints=expected if (expected[1]-expected[0])@direction>=0 else expected[::-1]
                        left=np.flatnonzero(np.linalg.norm(e['a']-endpoints[0],axis=1)<=3.)
                        right=np.flatnonzero(np.linalg.norm(e['b']-endpoints[1],axis=1)<=3.)
                        available_start|=bool(len(left));available_end|=bool(len(right))
                        for a in left:
                            for b in right:
                                if e['hi'][b]<=e['lo'][a]+1e-6:continue
                                pair=True;s=int(e['starts'][a]);t=int(e['ends'][b])
                                if t>=s and (e['prefix'][t+1]-e['prefix'][s])/(t-s+1)>=.5:accepted_pair=True
                counts['geometry_profile_available']+=has_profile;counts['selected_profile_available']+=has_selected
                counts['start_event_within3mm']+=available_start;counts['end_event_within3mm']+=available_end
                counts['both_events_on_one_profile']+=pair;counts['pair_also_passes_foreground_gate']+=accepted_pair
                children.append(dict(parent=pi,child=ci,kind=child['original_kind'],geometry_profile=has_profile,
                    selected_profile=has_selected,start=available_start,end=available_end,pair=pair,foreground_pair=accepted_pair))
        row=dict(source=si,**dict(counts));rows.append(row)
        write(run/f'source_{si:02d}_event_upper_bound.json',dict(scope='OFFLINE_NONEXCLUSIVE_EVENT_COVERAGE_NOT_ACCURACY',counts=row,children=children))
        print('EVENT_COVERAGE_UPPER_BOUND',row,flush=True)
    totals={k:sum(r[k] for r in rows) for k in rows[0] if k!='source'}
    write(run/'endpoint_event_audit_825.json',dict(scope='OFFLINE_NONEXCLUSIVE_EVENT_COVERAGE_NOT_ACCURACY',
        source_sha256=digest(__file__),rows=rows,totals=totals,
        GT_only_for_posthoc_audit=True,duplicate_capacity_not_enforced=True,no_geometry_or_weights_changed=True,
        all_496_including_curves_retained=True,full_contract_pass=False))
    print('EVENT_AUDIT_TOTAL',totals,flush=True)


if __name__=='__main__':main()
