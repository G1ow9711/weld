"""Raw source isolation plus coordinate equivalence for the current candidate."""
import argparse
import builtins
import io
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.predict_expanded_candidate_model import predict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--structured',action='store_true');args=parser.parse_args()
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    piece=load_piece(split['piece_roots'][args.name],verify=True);pcd=Path(piece['record']['input']).resolve()
    role='FIT' if args.name in split['train'] else 'UNSEEN_CHECK'
    geometry=Path('weld_pipeline/runs/contract_full_fit_hybrid_v1')
    expected=json.loads((geometry/(role+'_'+args.name+'.json')).read_text('utf8'))['predictions']
    run=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4' if args.structured else 'weld_pipeline/runs/contract_full_fit_hybrid_attributes_v3');out=run/('raw_'+args.name)
    forbidden=[Path(p).resolve() for p in ['weld_pipeline/data','weld_pipeline/reports']]
    allowed_reads=[];builtin_open,io_open,path_open=builtins.open,io.open,Path.open
    def check_read(file,mode):
        read=('r' in str(mode)) or ('+' in str(mode) and not str(mode).startswith(('w','x')))
        if isinstance(file,(str,bytes,Path)) and read:
            path=Path(file).resolve()
            if path==pcd:
                allowed_reads.append(str(path))
            elif any(path.is_relative_to(root) for root in forbidden) or path.suffix.lower() in ['.stp','.step','.xlsx','.xls','.cbf'] or path.name.endswith('_welds.json'):
                raise RuntimeError('Forbidden reference/cache read inside prediction: '+str(path))
    def guard(original):
        def checked(file,mode='r',*positional,**kwargs):
            check_read(file,mode)
            return original(file,mode,*positional,**kwargs)
        return checked
    def guarded_path(self,mode='r',*positional,**kwargs):
        check_read(self,mode)
        return path_open(self,mode,*positional,**kwargs)
    builtins.open,io.open,Path.open=guard(builtin_open),guard(io_open),guarded_path
    try:
        # Python 3.10 caches an accessor's io.open. Guard Path.open explicitly.
        try:
            (forbidden[0]/'read_guard_negative_control.stp').open('r')
        except RuntimeError:
            pass
        else:
            raise ValueError('Path.open reference guard failed its negative control')
        if args.structured:
            from weld_pipeline.predict_structured_candidate_model import predict as selected_predict
        else:
            selected_predict=predict
        pred,receipt=selected_predict(pcd,run/'best.pt',out)
    finally:
        builtins.open,io.open,Path.open=builtin_open,io_open,path_open
    if len(pred)!=len(expected):
        raise ValueError('Raw/cache prediction count mismatch')
    maximum=0.
    for got,wanted in zip(pred,expected):
        if got['type']!=wanted['type'] or len(got['segments'])!=len(wanted['segments']):
            raise ValueError('Raw/cache type or segmentation mismatch')
        for a,b in zip(got['segments'],wanted['segments']):
            qa,qb=np.array(a['points']),np.array(b['points'])
            if a['kind']!=b['kind'] or qa.shape!=qb.shape:
                raise ValueError('Raw/cache representation mismatch')
            maximum=max(maximum,float(np.abs(qa-qb).max()))
    if maximum>1e-6 or not allowed_reads:
        raise ValueError('Raw point provenance/equivalence failed')
    score=score_children(pred,piece['truth'])
    (out/'evaluation_score.json').write_text(json.dumps(score,indent=2),encoding='utf8')
    audit={'name':args.name,'role':role,'status':'PASS_RAW_REFERENCE_ISOLATION_AND_CACHED_EQUIVALENCE',
        'maximum_coordinate_difference_mm':maximum,'reference_cache_open_guard_active':True,
        'guarded_pcd_read_count':len(allowed_reads),'checkpoint_sha256':sha256(run/'best.pt'),
        'score':score,'full_contract_pass':False}
    (out/'raw_verification.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print('EXPANDED_RAW_VERIFIED',args.name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)


if __name__=='__main__':
    main()
