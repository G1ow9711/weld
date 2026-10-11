"""Full original teacher representation after bounded physical-cell encoding."""
import gc
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.bounded_event_phase import CELL_MM,cell_residuals,cell_offsets
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.continuous_center_modes import RESOLUTION_MM,center_modes,restore_event_order
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.train_forest_event_centers import load_cached

OUT=Path('weld_pipeline/reports/bounded_phase_centers_175_20261002')


def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')


def analytic(intervals,step,cuts=()):
    coordinates=np.arange(intervals[0][0]-20.,intervals[-1][1]+20.+step/2.,step)
    positions=np.arange(len(coordinates),dtype=float)*step
    seq=dict(xyz=np.c_[coordinates,np.zeros((len(coordinates),2))],direction=np.array([1.,0.,0.]),type='horizontal',sample_step_mm=step)
    foreground=np.zeros(len(coordinates));support=np.zeros((len(coordinates),3));offsets=np.zeros_like(support)
    for a,b in intervals:foreground[(coordinates>=a)&(coordinates<=b)]=1.
    for role,locations in enumerate(([a for a,b in intervals],[b for a,b in intervals],list(cuts))):
        if not locations:continue
        delta=np.asarray(locations)[None,:]-coordinates[:,None];nearest=np.argmin(np.abs(delta),axis=1)
        offset=delta[np.arange(len(delta)),nearest];support[:,role]=np.abs(offset)<=16.;offsets[:,role]=offset/16.
    residuals=cell_residuals(offsets,positions).astype(np.float32);recovered=cell_offsets(residuals,positions)
    events=center_modes(support,recovered,step);predictions=decode_event_sets([seq],[foreground],[events])
    actual=sorted((s['points'][0][0],s['points'][-1][0]) for w in predictions for s in w['segments'])
    expected=[]
    for a,b in intervals:
        divisions=[a]+[p for p in cuts if a<p<b]+[b];expected.extend(zip(divisions[:-1],divisions[1:]))
    passed=len(actual)==len(expected) and np.allclose(actual,expected,atol=RESOLUTION_MM,rtol=0.)
    return dict(intervals_mm=intervals,cuts_mm=list(cuts),sample_step_mm=step,
        recovered_children_mm=actual,topology_and_coordinates_retained=bool(passed))


def main():
    if OUT.exists():raise ValueError('Preserve bounded-cell representation proof')
    config=json.loads(Path('weld_pipeline/runs/forest_event_centers_seed166/config.json').read_text('utf8'))
    OUT.mkdir(parents=True)
    rows=[]
    for step in (.25,2.):
        for length in (.1,1.,2.5,3.,3.1,6.2680269):rows.append(analytic([(0.,length)],step))
        for gap in (.1,.5,1.,2.):rows.append(analytic([(0.,10.),(10.+gap,20.+gap)],step))
        rows.append(analytic([(0.,10.)],step,cuts=(1.,1.5,9.9)))
        rows.append(analytic([(63.8,65.3)],step))
    # Explicit cross-cell inverse and the -1 offset that remains a valid target.
    positions=np.array([47.8,63.8,64.,65.8,79.8]);offsets=np.broadcast_to(((63.8-positions)/16.)[:,None],(len(positions),3)).copy()
    residuals=cell_residuals(offsets,positions).astype(np.float32)
    inverse_error=float(np.abs(cell_offsets(residuals,positions)-offsets).max()*16.)
    dump(OUT/'analytic.json',dict(rows=rows,all_pass=all(r['topology_and_coordinates_retained'] for r in rows),
        cross_cell_and_negative_one_inverse_error_mm=inverse_error))
    if not all(r['topology_and_coordinates_retained'] for r in rows) or inverse_error>1e-5:
        raise ValueError('Bounded phase loses analytic topology or signed inverse')
    scores={};receipts={};max_error=0.
    for name in config['train']:
        data,near,known,piece=load_cached(name)
        foreground=[np.zeros(len(s['xyz']),np.float32) for s in piece['sequences']];events=[[[],[],[]] for s in piece['sequences']]
        for e in data['entries']:
            if e['type']!='horizontal':continue
            seq,_,tf=canonical_frame(piece['sequences'][e['sequence']]);a,b=e['start'],e['stop'];step=float(seq['sample_step_mm'])
            yy=np.asarray(near[a:b],float);kk=np.asarray(known[a:b]);positions=np.arange(len(yy),dtype=float)*step
            residuals=cell_residuals(yy[:,4:7],positions).astype(np.float32);recovered=cell_offsets(residuals,positions)
            valid=kk[:,4:7]>0
            if valid.any():max_error=max(max_error,float(np.abs(recovered-yy[:,4:7])[valid].max()*16.))
            hypotheses=center_modes(np.where(kk[:,1:4]>0,yy[:,1:4],0.),recovered,step)
            foreground[e['sequence']],events[e['sequence']]=restore_event_order(np.clip(yy[:,0],0.,1.),hypotheses,(len(yy)-1)*step,tf)
        predictions=decode_event_sets(piece['sequences'],foreground,events)
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        score=score_children(predictions,truth);scores[name]=score
        dump(OUT/(name+'.json'),dict(score=score,predictions=predictions,teacher_is_not_model_output=True,full_contract_pass=False))
        receipts[name]=sha256(OUT/(name+'.json'))
        print('BOUNDED_PHASE_ORACLE',name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
        del data,near,known,piece,foreground,events;gc.collect()
    total=aggregate(scores);passed=total['truth_count']==1221 and total['valid_children']==1221 and total['false_predictions']==0 and max_error<=1e-5
    dump(OUT/('complete.json' if passed else 'failure.json'),dict(teacher_score=total,
        all_original_1221_and_analytic_cases_represented=passed,analytic_cases=len(rows),cell_mm=CELL_MM,
        max_known_inverse_error_mm=max_error,piece_receipts=receipts,original_labels_or_weights_changed=False,
        source_sha256={__file__:sha256(__file__),'weld_pipeline/bounded_event_phase.py':sha256('weld_pipeline/bounded_event_phase.py')},
        actual_trained_models=0,full_contract_pass=False))
    print('BOUNDED_PHASE_ORACLE_COMPLETE',passed,total['valid_children'],'/',total['truth_count'],'INVERSE_MM',max_error,flush=True)
    if not passed:raise ValueError('Bounded phase does not represent complete source scope')


if __name__=='__main__':main()
