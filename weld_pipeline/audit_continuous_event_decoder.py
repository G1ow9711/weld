"""Teacher representation proof, not trained-model or customer acceptance."""
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.continuous_event_decoder import VERSION, decode_event_sets
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import event_peaks

OUT = Path('weld_pipeline/reports/continuous_event_decoder_169_20261002')
BASE = Path('weld_pipeline/data/joint_relation_v1')
RUN = Path('weld_pipeline/runs/forest_event_centers_seed166')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def analytic(intervals, step, cuts=(), offset=.0, reverse=False):
    coordinates = np.arange(-20., intervals[-1][1]+20.+step/2., step)
    direction = np.array([-1., 0., 0.]) if reverse else np.array([1., 0., 0.])
    origin = np.array([offset, 37., -11.])
    xyz = origin+coordinates[:, None]*direction
    seq = dict(xyz=xyz, direction=direction, type='horizontal', sample_step_mm=step)
    foreground = np.zeros(len(coordinates))
    for a,b in intervals:foreground[(coordinates >= a) & (coordinates <= b)] = 1.
    events = [[[x-coordinates[0], .9999] for x in role] for role in
              ([a for a,b in intervals], [b for a,b in intervals], list(cuts))]
    predictions = decode_event_sets([seq], [foreground], [events])
    truth = []
    for i,(a,b) in enumerate(intervals):
        divisions = [a]+[x for x in cuts if a < x < b]+[b]
        truth.append(dict(id=f'analytic_{i}', type='horizontal', segments=[dict(kind='line',
            points=[(origin+u*direction).tolist(), (origin+v*direction).tolist()]) for u,v in zip(divisions[:-1],divisions[1:])]))
    score = score_children(predictions, truth)
    actual = sorted((float(np.dot(np.asarray(s['points'][0])-origin, direction)),
                     float(np.dot(np.asarray(s['points'][-1])-origin, direction)))
                    for w in predictions for s in w['segments'])
    expected = sorted((float(np.dot(np.asarray(s['points'][0])-origin, direction)),
                       float(np.dot(np.asarray(s['points'][-1])-origin, direction)))
                      for w in truth for s in w['segments'])
    exact = len(actual) == len(expected) and np.allclose(actual, expected, atol=1e-9, rtol=0.)
    return dict(intervals_mm=intervals, cuts_mm=list(cuts), step_mm=step, reverse=reverse,
        translated_origin_x=offset, decoded_children_mm=actual, exact_topology_and_coordinates=bool(exact), score=score)


def main():
    if OUT.exists():raise ValueError('Preserve the fixed continuous-event proof')
    config = json.loads((RUN/'config.json').read_text('utf8'))
    OUT.mkdir(parents=True); snapshot=OUT/'source_snapshot';snapshot.mkdir()
    sources = [__file__, 'weld_pipeline/continuous_event_decoder.py', 'weld_pipeline/event_gap_decoder.py',
               'weld_pipeline/contract_child_evaluation.py']
    for path in sources:shutil.copy2(path, snapshot/Path(path).name)
    dump(OUT/'protocol.json', dict(version=VERSION, source_sha256={p:sha256(p) for p in sources},
        sources=config['train'], teacher_event_sets_not_runtime_predictions=True,
        expected_truth_children=1221, original_targets_unchanged=True, frozen_166_source_unchanged=True,
        analytic_cases_are_not_original_measured_dataset=True, full_contract_pass=False))
    began=time.perf_counter(); rows=[]
    for step in (.25, 2.):
        for length in (.1, 1., 2.5, 3., 3.1, 6.2680269):
            rows.append(analytic([(0.,length)], step))
        for gap in (.1,.5,1.,2.):
            rows.append(analytic([(0.,10.),(10.+gap,20.+gap)],step))
        rows.append(analytic([(0.,10.)],step,cuts=(1.,1.5,9.9)))
        rows.append(analytic([(.5,1.5)],step,offset=16735.42,reverse=True))
    dump(OUT/'analytic.json', dict(rows=rows, all_exact=all(r['exact_topology_and_coordinates'] for r in rows)))
    if not all(r['exact_topology_and_coordinates'] for r in rows):
        dump(OUT/'failure.json', dict(stage='analytic', failed_indices=[i for i,r in enumerate(rows) if not r['exact_topology_and_coordinates']], full_contract_pass=False))
        raise ValueError('Analytic short/gap topology not retained')
    scores={}; receipts={}
    for name in config['train']:
        manifest_path=BASE/name/'manifest.json';manifest=json.loads(manifest_path.read_text('utf8'))
        piece=load_piece(manifest['source_cache_root'], verify=True)
        foreground=[];events=[]
        for seq,target in zip(piece['sequences'],piece['targets']):
            if seq['type'] != 'horizontal' or seq.get('curve_kind') == 'arc':
                foreground.append(np.zeros(len(seq['xyz'])));events.append([[],[],[]]);continue
            foreground.append(np.clip(target[:,0],0.,1.))
            step=float(seq['sample_step_mm'])
            events.append([[[p*step,h] for p,h in event_peaks(np.maximum(target[:,role],0),step,.5)] for role in (1,2,3)])
        predictions=decode_event_sets(piece['sequences'],foreground,events)
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        score=score_children(predictions,truth);scores[name]=score
        dump(OUT/(name+'.json'),dict(score=score,predictions=predictions,base_manifest_sha256=sha256(manifest_path),
            teacher_events_not_model_outputs=True,full_contract_pass=False))
        receipts[name]=sha256(OUT/(name+'.json'))
        print('CONTINUOUS_EVENT_ORACLE',name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
        del piece,foreground,events,predictions;gc.collect()
    total=aggregate(scores)
    passed=total['truth_count']==1221 and total['valid_children']==1221 and total['false_predictions']==0
    result=dict(protocol_sha256=sha256(OUT/'protocol.json'), analytic_cases=len(rows),
        analytic_all_topologies_and_coordinates_exact=True, teacher_fit_score=total,
        all_1221_original_children_represented=passed,piece_receipts=receipts,
        decoder_adopted=False,actual_trained_models=0,elapsed_seconds=time.perf_counter()-began,full_contract_pass=False)
    dump(OUT/('complete.json' if passed else 'failure.json'), result)
    print('CONTINUOUS_EVENT_ORACLE_COMPLETE',passed,total['valid_children'],'/',total['truth_count'],'FP',total['false_predictions'],flush=True)
    if not passed:raise ValueError('Full unchanged FIT teacher scope not represented')


if __name__=='__main__':main()
