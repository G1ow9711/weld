"""Check actual H context against its training cache and existing V context."""
import json
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.fresh_point_context import point_structure
from weld_pipeline.full_fit_hybrid import point_structure as vertical_context


def main():
    path=Path('weld_pipeline/reports/fresh_point_context_20261001.json')
    if path.exists():
        raise ValueError('Frozen all-contact context check exists')
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    rows=[]
    for name in ['207-FR79M','224-GR24A1','203-TT1A']:
        piece=load_piece(split['piece_roots'][name],verify=True)
        xyz=read_xyz_pcd(piece['record']['input'])*1000.
        actual,process=point_structure(xyz,piece['sequences'])
        old,old_process=vertical_context(xyz,piece['sequences'])
        cache=Path(split['structure_roots'][name])/(name+'.npz')
        with np.load(cache,allow_pickle=False) as source:
            expected,expected_process=source['structural'].copy(),source['process'].copy()
        h=np.array([s['type']=='horizontal' and s.get('curve_kind')!='arc' for s in piece['sequences']])
        v=np.array([s['type']=='vertical' for s in piece['sequences']])
        h_error=float(np.max(np.abs(actual[h]-expected[h]))) if h.any() else 0.
        hp_error=float(np.max(np.abs(process[h]-expected_process[h]))) if h.any() else 0.
        v_error=float(np.max(np.abs(actual[v]-old[v]))) if v.any() else 0.
        vp_error=float(np.max(np.abs(process[v]-old_process[v]))) if v.any() else 0.
        rows.append(dict(piece=name,horizontal_contacts=int(h.sum()),vertical_contacts=int(v.sum()),
            H_structure_max_error_to_training_cache=h_error,H_sizes_max_error_to_training_cache=hp_error,
            V_structure_max_error_to_frozen_helper=v_error,V_sizes_max_error_to_frozen_helper=vp_error,
            cached_source_sha256=sha256(cache),input_sha256=sha256(piece['record']['input']),
            H_context_was_zero_in_old_helper=bool(np.count_nonzero(old[h])==0),
            H_context_now_populated=bool(np.count_nonzero(actual[h])>0),
            H_cache_equal=bool(np.array_equal(actual[h],expected[h]) and np.array_equal(process[h],expected_process[h])),
            V_frozen_helper_equal=bool(np.array_equal(actual[v],old[v]) and np.array_equal(process[v],old_process[v]))))
        print('FRESH_POINT_CONTEXT',name,'H_ERROR',h_error,hp_error,'V_ERROR',v_error,vp_error,flush=True)
    result=dict(cases=rows,passed=all(r['H_cache_equal'] and r['V_frozen_helper_equal'] for r in rows),
        source_sha256=sha256('weld_pipeline/fresh_point_context.py'),verifier_sha256=sha256(__file__),
        check_only_no_acceptance_claim=True,observed_check_read=False,full_contract_pass=False)
    path.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('FRESH_CONTEXT_CHECK_COMPLETE',result['passed'],flush=True)


if __name__=='__main__':
    main()
