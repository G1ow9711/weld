"""Read-only capacity diagnosis on trained sources of fixed cross-fit models."""
import json
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.joint_relation_heatmaps import JointRelationHeatmaps,infer_cached
from weld_pipeline.train_joint_relation_crossfit import DATA,OUT,load_data


def main():
    out=Path('weld_pipeline/reports/joint_relation_fit_capacity_20261001_v2')
    if out.exists():
        raise ValueError('Frozen FIT capacity audit exists')
    out.mkdir(parents=True)
    protocol=json.loads((OUT/'config.json').read_text('utf8'))
    complete=json.loads((OUT/'complete.json').read_text('utf8'))
    fold=protocol['folds'][0]
    config=json.loads((DATA/'config.json').read_text('utf8'));config['pieces']=fold['train_real']
    data=load_data(config);torch.set_num_threads(2);results={}
    for messages in (False,True):
        kind='messages' if messages else 'no_messages'
        run=OUT/('fold0_seed143_'+kind);weights=run/'final.pt';frozen=sha256(weights)
        state=torch.load(weights,map_location='cpu');model=JointRelationHeatmaps(messages).cuda()
        model.load_state_dict(state['model']);scores={};details={}
        for name in fold['train_real']:
            piece=load_piece(data[name]['manifest']['source_cache_root'],verify=True)
            probabilities=infer_cached(model,data[name],piece['sequences'])
            predictions=decode_intervals(piece['sequences'],probabilities,threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
            truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
            score=score_children(predictions,truth);scores[name]=score
            heat=np.zeros((4,4),np.int64)
            for sequence,target,p in zip(piece['sequences'],piece['targets'],probabilities):
                if sequence['type']!='horizontal' or sequence.get('curve_kind')=='arc':
                    continue
                y=target[:,:4]
                for column in range(4):
                    positive=y[:,column]>(.5 if column==0 else .65)
                    background=(y[:,column]>=0)&(y[:,column]<=(.5 if column==0 else .001))
                    heat[column]+=[positive.sum(),((p[:,column]>.5)&positive).sum(),
                                   background.sum(),((p[:,column]>.5)&background).sum()]
            details[name]=dict(score=score,heatmap_counts_positive_detected_background_false=heat.tolist(),predictions=predictions)
            print('RELATION_CAPACITY',kind,name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
        if sha256(weights)!=frozen:
            raise ValueError('Fixed weights changed in FIT audit')
        results[kind]=dict(summary=aggregate(scores),per_source=details,weights_sha256=frozen)
        del model;torch.cuda.empty_cache()
    result=dict(results=results,fold=0,seed=143,role='TRAINED_SOURCE_CAPACITY_ONLY',
        fixed_crossfit_adoption_eligible=complete['message_adoption_eligible'],
        no_weights_or_decoder_changed=True,held_or_observed_not_used_to_fit=True,
        source_sha256=sha256(__file__),full_contract_pass=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('RELATION_CAPACITY_COMPLETE',json.dumps({key:value['summary'] for key,value in results.items()}),flush=True)


if __name__=='__main__':
    main()
