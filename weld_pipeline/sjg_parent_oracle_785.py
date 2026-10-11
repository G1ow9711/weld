"""Diagnostic only: distinguish wrong grouping from incomplete predicted parents.

Uses target identities to build an oracle. Outputs must never be deployed or
reported as model predictions. All existing predicted children remain counted.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.sjg_geometry_776 import read, write
from weld_pipeline.sjg_prepare_776 import ROOT, OUT, TRAIN
from weld_pipeline.sjg_path_examples_784 import frozen_prediction
from weld_pipeline.sjg_path_781 import free_children
from weld_pipeline.sjg_contract_score_778 import adapt, total
from weld_pipeline.contract_geometry_d3_409 import score_source


def main():
    inventory = read(OUT/'inventory.json'); scores=[]; rows=[]
    for si in TRAIN:
        contacts = read(OUT/f'source_{si:02d}/geometry.json')['contacts']
        children = free_children(frozen_prediction(si, contacts), contacts)
        coverage = read(OUT/f'source_{si:02d}/coverage_v2.json')['assignments']
        target = read(ROOT/inventory[si]['target']['target_path'])
        labels = {}
        for ci in {c['contact'] for c in children}:
            ids=[i for i,c in enumerate(children) if c['contact']==ci]
            truth=[a for a in coverage if a['contact']==ci]
            if not truth: continue
            actual=np.array([[children[i]['lo'],children[i]['hi']] for i in ids])
            expected=np.array([[a['lo'],a['hi']] for a in truth])
            error=np.abs(actual[:,None]-expected[None]).max(2)
            cost=np.c_[np.where(error<=3,error,1e6),np.full((len(ids),len(ids)),1e4)]
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                if j<len(truth) and error[i,j]<=3: labels[ids[i]]=truth[j]
        groups={}; unmatched=[]
        for i,c in enumerate(children):
            if i in labels: groups.setdefault(labels[i]['parent'],[]).append((labels[i]['child'],c))
            else: unmatched.append(c)
        output=[]; incomplete=0; complete=0
        for parent, group in groups.items():
            group.sort(key=lambda v:v[0]); ref=target['parents'][parent]
            if len(group)==len(ref['children']): complete+=1
            else: incomplete+=1
            segments=[]
            for ci,c in group:
                points=np.array([c['start'],c['end']]); truth=np.array(ref['children'][ci]['world_representation']['endpoints_mm'])
                if np.linalg.norm(points-truth)>np.linalg.norm(points[::-1]-truth): points=points[::-1].copy()
                segments.append(dict(kind='line',points=points.tolist()))
            for k in range(len(group)-1):
                first,second=group[k][0],group[k+1][0]
                if second==first+1 and ref['gaps'][first]=='SHARED':
                    shared=((np.array(segments[k]['points'][-1])+segments[k+1]['points'][0])*.5).tolist()
                    segments[k]['points'][-1]=shared;segments[k+1]['points'][0]=shared
            output.append(dict(id=str(len(output)),type=group[0][1]['weld_type'],weld_side=None,segments=segments))
        for c in unmatched:
            output.append(dict(id=str(len(output)),type=c['weld_type'],weld_side=None,
                segments=[dict(kind='line',points=np.array([c['start'],c['end']]).tolist())]))
        score=score_source(adapt(output),target);scores.append(score)
        current=read(OUT/f'run09/source_{si:02d}_contract_score.json')['categories']['overall']
        row=dict(source=si,actual=current,oracle=score['categories']['overall'],
                 complete_matched_parents=complete,incomplete_matched_parents=incomplete)
        rows.append(row);print(row,flush=True)
    write(OUT/'run09/parent_oracle_diagnostic.json',dict(rows=rows,oracle_total=total(scores),
          uses_GT_parent_identity=True,not_model_accuracy=True,no_optimizer_updates=True))


if __name__=='__main__':main()
