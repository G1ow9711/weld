"""Coupled physical-center representation proof on full FIT and analytic cases."""
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.audit_continuous_event_decoder import analytic
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.continuous_center_modes import VERSION,RESOLUTION_MM,center_modes,restore_event_order
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.train_forest_event_centers import load_cached

OUT=Path('weld_pipeline/reports/direct_center_modes_171_20261002')


def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')


def mode_case(intervals,step,cuts=()):
    coordinates=np.arange(-20.,intervals[-1][1]+20.+step/2.,step)
    xyz=np.c_[coordinates,np.zeros((len(coordinates),2))]
    seq=dict(xyz=xyz,direction=np.array([1.,0.,0.]),sample_step_mm=step,type='horizontal')
    foreground=np.zeros(len(coordinates));support=np.zeros((len(coordinates),3));offsets=np.zeros_like(support)
    for a,b in intervals:foreground[(coordinates>=a)&(coordinates<=b)]=1.
    for role,positions in enumerate(([a for a,b in intervals],[b for a,b in intervals],list(cuts))):
        if not positions:continue
        delta=np.asarray(positions)[None,:]-coordinates[:,None]
        nearest=np.argmin(np.abs(delta),axis=1);offset=delta[np.arange(len(delta)),nearest]
        support[:,role]=np.abs(offset)<=16.;offsets[:,role]=offset/16.
    events=center_modes(support,offsets,step);predictions=decode_event_sets([seq],[foreground],[events])
    actual=sorted((s['points'][0][0],s['points'][-1][0]) for w in predictions for s in w['segments'])
    expected=[]
    for a,b in intervals:
        divisions=[a]+[p for p in cuts if a<p<b]+[b];expected.extend(zip(divisions[:-1],divisions[1:]))
    exact=len(actual)==len(expected) and np.allclose(actual,expected,atol=RESOLUTION_MM,rtol=0.)
    return dict(intervals_mm=intervals,cuts_mm=cuts,sample_step_mm=step,actual_children_mm=actual,
        centers_from_query_offsets=True,topology_and_coordinates_retained=bool(exact))


def main():
    if OUT.exists():raise ValueError('Preserve the direct mode proof')
    config=json.loads(Path('weld_pipeline/runs/forest_event_centers_seed166/config.json').read_text('utf8'))
    OUT.mkdir(parents=True);snapshot=OUT/'source_snapshot';snapshot.mkdir()
    sources=[__file__,'weld_pipeline/continuous_center_modes.py','weld_pipeline/continuous_event_decoder.py']
    for p in sources:shutil.copy2(p,snapshot/Path(p).name)
    dump(OUT/'protocol.json',dict(version=VERSION,numerical_position_resolution_mm=RESOLUTION_MM,
        source_sha256={p:sha256(p) for p in sources},sources=config['train'],
        teachers_only_not_model_scores=True,expected_truth_children=1221,full_contract_pass=False))
    rows=[]
    for step in (.25,2.):
        for length in (.1,1.,2.5,3.,3.1,6.2680269):rows.append(mode_case([(0.,length)],step))
        for gap in (.1,.5,1.,2.):rows.append(mode_case([(0.,10.),(10.+gap,20.+gap)],step))
        rows.append(mode_case([(0.,10.)],step,cuts=(1.,1.5,9.9)))
        rows.append(mode_case([(.5,1.5)],step))
    dump(OUT/'analytic.json',dict(rows=rows,all_pass=all(r['topology_and_coordinates_retained'] for r in rows)))
    if not all(r['topology_and_coordinates_retained'] for r in rows):
        dump(OUT/'failure.json',dict(stage='analytic',failed_indices=[i for i,r in enumerate(rows) if not r['topology_and_coordinates_retained']],full_contract_pass=False))
        raise ValueError('Direct physical event fields lose analytic topology')
    began=time.perf_counter();scores={};receipts={}
    for name in config['train']:
        data,near,known,piece=load_cached(name)
        foregrounds=[np.zeros(len(s['xyz']),np.float32) for s in piece['sequences']];event_sets=[[[],[],[]] for s in piece['sequences']]
        for e in data['entries']:
            if e['type']!='horizontal':continue
            seq,_,transform=canonical_frame(piece['sequences'][e['sequence']]);a,b=e['start'],e['stop']
            yy=np.asarray(near[a:b],float);kk=np.asarray(known[a:b]);step=float(seq['sample_step_mm'])
            events=center_modes(np.where(kk[:,1:4]>0,yy[:,1:4],0.),yy[:,4:7],step)
            foregrounds[e['sequence']],event_sets[e['sequence']]=restore_event_order(np.clip(yy[:,0],0.,1.),events,(len(seq['xyz'])-1)*step,transform)
        predictions=decode_event_sets(piece['sequences'],foregrounds,event_sets)
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        score=score_children(predictions,truth);scores[name]=score
        dump(OUT/(name+'.json'),dict(score=score,predictions=predictions,full_contract_pass=False))
        receipts[name]=sha256(OUT/(name+'.json'))
        print('DIRECT_CENTER_ORACLE',name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
        del data,near,known,piece,foregrounds,event_sets;gc.collect()
    total=aggregate(scores);passed=total['truth_count']==1221 and total['valid_children']==1221 and total['false_predictions']==0
    dump(OUT/('complete.json' if passed else 'failure.json'),dict(oracle_score=total,
        original_1221_and_analytic_cases_represented=passed,analytic_cases=len(rows),piece_receipts=receipts,
        numerical_position_resolution_mm=RESOLUTION_MM,decoder_adopted=False,actual_trained_models=0,
        elapsed_seconds=time.perf_counter()-began,full_contract_pass=False))
    print('DIRECT_CENTER_ORACLE_COMPLETE',passed,total['valid_children'],'/',total['truth_count'],'FP',total['false_predictions'],flush=True)
    if not passed:raise ValueError('Direct centers lose complete unchanged source scope')


if __name__=='__main__':main()
