"""Prove the new event representation before any gradient training or held read."""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals,event_peaks
from weld_pipeline.joint_event_votes import dense_targets,vote_heatmaps,RADIUS_MM,SIGMA_MM,VOTE_SCALE,TAIL_MM

OUT=Path('weld_pipeline/reports/joint_event_vote_targets_151_20261001')


def main():
    if OUT.exists():raise ValueError('Preserve event-vote representation proof')
    OUT.mkdir(parents=True)
    cases=[];step=2.;positions=np.arange(121)*step
    for label,events in [('no_events',[]),('subsample',[37.3]),('near_boundary',[2.7,235.1]),
                         ('two_near',[67.3,75.3]),('multiple',[33.2,88.7,165.1,217.8])]:
        target=np.zeros((len(positions),4),np.float32)
        for event in events:target[:,1]=np.maximum(target[:,1],np.exp(-.5*((positions-event)/1.5)**2))
        dense=dense_targets(target,step)
        result=vote_heatmaps(dense[:,0],dense[:,1:4],dense[:,4:7],step)
        got=[p*step for p,_ in event_peaks(result[:,1],step,.5)]
        error=max((min(abs(e-g) for g in got) if got else float('inf') for e in events),default=0.)
        if len(got)!=len(events) or error>3.:raise ValueError('Physical event-vote representation failed: '+label)
        reversed_target=target[::-1].copy();reversed_target[:,[1,2]]=reversed_target[:,[2,1]]
        reversed_dense=dense_targets(reversed_target,step)
        reversed_p=vote_heatmaps(reversed_dense[:,0],reversed_dense[:,1:4],reversed_dense[:,4:7],step)
        expected=result[::-1].copy();expected[:,[1,2]]=expected[:,[2,1]]
        reverse_error=float(np.abs(reversed_p-expected).max())
        if reverse_error>2e-6:raise ValueError('Dense event-vote reversal failed')
        cases.append(dict(case=label,event_count=len(events),max_event_error_mm=error,reverse_error=reverse_error,passed=True))
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    data_config_path=Path('weld_pipeline/data/joint_relation_v1/config.json')
    data_config=json.loads(data_config_path.read_text('utf8'));names=data_config['folds'][0]['train_real']
    scores={}
    for name in names:
        piece=load_piece(split['piece_roots'][name],verify=True)
        probabilities=[]
        for sequence,target in zip(piece['sequences'],piece['targets']):
            p=np.zeros((len(target),7),np.float32)
            if sequence['type']=='horizontal' and sequence.get('curve_kind')!='arc':
                canonical,labels,transform=canonical_frame(sequence,target)
                dense=dense_targets(labels,sequence['sample_step_mm'])
                p=restore_probability_order(vote_heatmaps(dense[:,0],dense[:,1:4],dense[:,4:7],sequence['sample_step_mm']),transform)
            probabilities.append(p)
        predictions=decode_intervals(piece['sequences'],probabilities,threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        score=score_children(predictions,truth);scores[name]=score
        (OUT/(name+'.json')).write_text(json.dumps(dict(score=score,predictions=predictions),indent=2),encoding='utf8')
        print('EVENT_VOTE_TARGET_REPRESENTATION',name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
    result=aggregate(scores)
    passed=result['valid_children']==result['truth_count'] and result['false_predictions']==0
    complete=dict(physical_cases=cases,fit_target_representation=result,all_target_checks_passed=passed,
        model_trained=False,training_allowed=passed,train_sources=names,
        source_sha256={p:sha256(p) for p in (__file__,'weld_pipeline/joint_event_votes.py','weld_pipeline/event_gap_decoder.py')},
        parameters=dict(radius_mm=RADIUS_MM,sigma_mm=SIGMA_MM,tail_mm=TAIL_MM,vote_scale=VOTE_SCALE),
        data_config_sha256=sha256(data_config_path),held_sources_read=False,observed_check_read=False,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')
    print('EVENT_VOTE_REPRESENTATION_COMPLETE',passed,result['valid_children'],result['false_predictions'],flush=True)


if __name__=='__main__':
    main()
