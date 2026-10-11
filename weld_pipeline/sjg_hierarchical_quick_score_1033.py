"""Independent scoring of already frozen hierarchical predictions."""
import argparse
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_local_event_score_932 import geometric_match

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--source',required=True);a=ap.parse_args()
    folder=OUT/a.run/('source_'+a.source);freeze=read(folder/'prediction_freeze.json')
    assert digest(folder/'predictions.json')==freeze['predictions_sha256']
    assert freeze['target_reads']==0
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==a.source)
    assert digest(OUT/case['target_file'])==case['target_sha256']
    target=read(OUT/case['target_file']);predictions=read(folder/'predictions.json')
    score=geometric_match(predictions,target);assert score['G']==case['original_G']
    children=[c for p in target['parents'] for c in p['children']]
    types=[]
    for m in score['matches']:
        expected=children[m['target']].get('weld_type')
        types.append(dict(target=m['target'],expected=expected,predicted=predictions[m['prediction']].get('weld_type'),
                          assessed=expected is not None,passed=expected is not None and expected==predictions[m['prediction']].get('weld_type')))
    score.update(source=a.source,role=case['role'],predictions_sha256=freeze['predictions_sha256'],target_sha256=case['target_sha256'],
                 elapsed_seconds=freeze['elapsed_seconds'],type_checks=types,full_contract_pass=False,end_utc=utc())
    score['maximum_endpoint_error_mm']=max((max(m['endpoint_errors_mm']) for m in score['matches']),default=None)
    score['maximum_bidirectional_error_mm']=max((m['bidirectional_sampled_maximum_mm'] for m in score['matches']),default=None)
    write(folder/'geometry_score.json',score)
    print('QUICK_SCORE',a.source,'TP',score['TP'],'G',score['G'],'FP',score['FP'],'MAX_MM',score['maximum_bidirectional_error_mm'],flush=True)

if __name__=='__main__':main()
