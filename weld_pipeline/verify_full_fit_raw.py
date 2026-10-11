"""Guard raw inference against reference/cache reads, then evaluate separately."""
import argparse
import builtins
import io
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.predict_full_fit_hybrid import predict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--attribute-candidates',action='store_true');parser.add_argument('--attribute-model',action='store_true')
    args=parser.parse_args()
    dataset=Path('weld_pipeline/data/full_available_fit_v1')
    split=json.loads((dataset/'split.json').read_text('utf8'))
    if args.name not in split['piece_roots']:
        raise ValueError('Unknown audit piece')
    piece=load_piece(split['piece_roots'][args.name],verify=True)
    pcd=Path(piece['record']['input']).resolve()
    role='FIT' if args.name in split['train'] else 'UNSEEN_CHECK'
    geometry_run=Path('weld_pipeline/runs/contract_full_fit_hybrid_v1')
    run=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v2') if args.attribute_model else geometry_run
    expected=json.loads((geometry_run/(role+'_'+args.name+'.json')).read_text('utf8'))['predictions']
    destination=run/('raw_'+args.name)
    cache_root=(Path('weld_pipeline/data')).resolve()
    report_root=(Path('weld_pipeline/reports')).resolve()
    guarded_reads=[]
    original_builtin,original_io=builtins.open,io.open
    def guarded(original):
        def open_file(file,mode='r',*positional,**kwargs):
            existing_read=('r' in str(mode)) or ('+' in str(mode) and not str(mode).startswith(('w','x')))
            if isinstance(file,(str,bytes,Path)) and existing_read:
                path=Path(file).resolve()
                if path==pcd:
                    guarded_reads.append(str(path))
                elif path.is_relative_to(cache_root) or path.is_relative_to(report_root) or path.suffix.lower() in ['.stp','.step','.xlsx','.xls','.cbf'] or path.name.endswith('_welds.json'):
                    raise RuntimeError('Reference/cache read forbidden inside raw prediction: '+str(path))
            return original(file,mode,*positional,**kwargs)
        return open_file
    builtins.open,io.open=guarded(original_builtin),guarded(original_io)
    try:
        if args.attribute_model:
            from weld_pipeline.predict_candidate_model import predict as predict_attributes
            pred,receipt=predict_attributes(pcd,run/'best.pt',destination)
        else:
            pred,receipt=predict(pcd,run/'best.pt',destination,args.attribute_candidates)
    finally:
        builtins.open,io.open=original_builtin,original_io
    if len(pred)!=len(expected):
        raise ValueError('Raw/cache parent count differs')
    maximum=0.
    for a,b in zip(pred,expected):
        if a['type']!=b['type'] or len(a['segments'])!=len(b['segments']):
            raise ValueError('Raw/cache type/topology differs')
        for sa,sb in zip(a['segments'],b['segments']):
            pa,pb=np.array(sa['points']),np.array(sb['points'])
            if sa['kind']!=sb['kind'] or pa.shape!=pb.shape:
                raise ValueError('Raw/cache curve representation differs')
            maximum=max(maximum,float(np.abs(pa-pb).max()))
    if maximum>1e-6:
        raise ValueError('Raw/cache point coordinates differ by '+str(maximum))
    score=score_children(pred,piece['truth'])
    (destination/'evaluation_score.json').write_text(json.dumps(score,indent=2),encoding='utf8')
    audit={'status':'PASS_RAW_REFERENCE_ISOLATION_AND_CACHED_EQUIVALENCE',
           'role':role,'name':args.name,'maximum_coordinate_difference_mm':maximum,
           'reference_cache_open_guard_active':True,'guarded_pcd_read_count':len(guarded_reads),
           'checkpoint_sha256':sha256(run/'best.pt'),'score':score,'full_contract_pass':False}
    (destination/'raw_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print('RAW_VERIFIED',args.name,role,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)


if __name__=='__main__':
    main()
