"""Posthoc distinction between weak event scores and missing event locations."""
import argparse
from collections import Counter
import numpy as np
from scipy.signal import find_peaks
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);base=OUT/'run53/free';inventory=read(OUT/'inventory.json');rows=[]
    write(dest/'registered_protocol.json',dict(stage='OFFLINE_SCORE_VERSUS_LOCATION_DIAGNOSIS',frozen_fields='run53/free',
        no_training=True,no_threshold_selection=True,no_modified_predictions=True,original_G=496,
        source_sha256=digest(__file__),full_contract_pass=False))
    for source in TRAIN+VALID+ARC:
        profiles=read(OUT/f'run34/source_{source:02d}/profiles.json');fields={}
        for item in read(base/f'source_{source:02d}_fields.json')['files']:
            path=base/item['file'];assert digest(path)==item['sha256']
            with np.load(path) as z:q=z['q_mm'];p=z['probability'];offset=z['offset']
            gap=max(1,int(np.ceil(2/max(float(q[1]-q[0]),1e-6))))
            heads=[]
            for head in (0,1):
                ids=find_peaks(np.r_[-1.,p[:,head+1],-1.],distance=gap)[0]-1
                refined=np.clip(q[ids]+np.clip(2*offset[ids,head],-4,4),q[0],q[-1])
                profile=profiles[item['profile']]
                points=np.asarray(profile['p0'])+refined[:,None]*profile['direction']
                heads.append(dict(ids=ids,points=points,probability=p[ids,head+1],refined=refined))
            fields[item['profile']]=dict(heads=heads,prefix=np.r_[0.,np.cumsum(p[:,0],dtype=np.float64)])
        target_path=ROOT/inventory[source]['target']['target_path'];assert digest(target_path)==inventory[source]['target']['target_sha256']
        target=read(target_path);supervision=read(OUT/f'run35/source_{source:02d}_supervision.json')
        options={(r['parent'],r['child']):r['profiles'] for r in supervision['children']};children=[];counts=Counter()
        for pi,parent in enumerate(target['parents']):
            for ci,child in enumerate(parent['children']):
                best=None;best_fg=None;selected=False
                if child['original_kind']=='line':
                    expected=np.asarray(child['world_representation']['endpoints_mm'])
                    for ni in options.get((pi,ci),[]):
                        if ni not in fields:continue
                        selected=True;direction=np.asarray(profiles[ni]['direction']);ends=expected if (expected[1]-expected[0])@direction>=0 else expected[::-1]
                        e=fields[ni];a,b=e['heads'];left=np.flatnonzero(np.linalg.norm(a['points']-ends[0],axis=1)<=3)
                        right=np.flatnonzero(np.linalg.norm(b['points']-ends[1],axis=1)<=3)
                        for i in left:
                            for j in right:
                                s=int(a['ids'][i]);t=int(b['ids'][j])
                                if t<s or b['refined'][j]<=a['refined'][i]+1e-6:continue
                                value=float(min(a['probability'][i],b['probability'][j]));best=value if best is None else max(best,value)
                                if (e['prefix'][t+1]-e['prefix'][s])/(t-s+1)>=.5:
                                    best_fg=value if best_fg is None else max(best_fg,value)
                status=('NONLINEAR_NOT_EXPRESSED' if child['original_kind']!='line' else
                    'NO_SELECTED_COMPATIBLE_PROFILE' if not selected else
                    'NO_LOCATION_COMPATIBLE_PEAK_PAIR' if best is None else
                    'PAIR_FOREGROUND_GATE_FAIL' if best_fg is None else
                    'WEAK_EVENT_SCORE_PAIR' if best_fg<.5 else 'BASELINE_EVENT_PAIR_PRESENT')
                counts[status]+=1;children.append(dict(parent=pi,child=ci,status=status,best_min_peak_probability=best,
                    best_min_peak_probability_with_foreground=best_fg))
        row=dict(source=source,G=len(children),counts=dict(counts));rows.append(row)
        write(dest/f'source_{source:02d}_score_location.json',dict(**row,children=children))
        print('SCORE_LOCATION_DIAGNOSIS',row,flush=True)
    write(dest/'receipt.json',dict(status='FROZEN_EVENT_SCORE_LOCATION_AUDITED',rows=rows,original_G=sum(r['G'] for r in rows),
        nonexclusive_geometric_event_upper_bound_not_weld_accuracy=True,no_threshold_selected=True,full_contract_pass=False))


if __name__=='__main__':main()
