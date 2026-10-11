"""Raw point generation followed by bitwise comparison with frozen features."""
import gc,json,time
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.fast_mixed_sequences import make_fast_mixed_sequences
from weld_pipeline.curved_cache import load_piece


def main():
    out=Path('weld_pipeline/reports/fast_mixed_equivalence');out.mkdir(exist_ok=True)
    records=[]
    for name in ['TD2-LB8A','527-FR146A','222-TT1A','527-LB24A1','234-FR71A']:
        folder=Path('data/supplemental' if name=='TD2-LB8A' else 'DataSet/9月20日')/name
        pcd=next(folder.glob('*AllFaces.pcd'));xyz=read_xyz_pcd(pcd)*1000
        start=time.perf_counter();actual,stats=make_fast_mixed_sequences(xyz);seconds=time.perf_counter()-start
        # Reference cache is accessed after point-only feature computation.
        cache=Path('weld_pipeline/data/mixed_surface_v1')/name;expected=load_piece(cache)['sequences']
        if len(actual)!=len(expected):raise ValueError('Sequence count differs: '+name)
        for index,(a,b) in enumerate(zip(actual,expected)):
            if not np.array_equal(a['xyz'],b['xyz']) or not np.array_equal(a['features'],b['features']):
                raise ValueError('Coordinate/feature mismatch: '+name+' sequence '+str(index))
            for key in ['type','sample_step_mm','curve_kind','curve','candidate_id']:
                if a.get(key)!=b.get(key):raise ValueError('Metadata differs: '+key)
            if not np.array_equal(a['direction'],b['direction']):raise ValueError('Direction differs')
        record={'group':name,'pcd_sha256':sha256(pcd),'manifest_sha256':sha256(cache/'manifest.json'),
                'sequence_count':len(actual),'feature_points':sum(len(s['xyz']) for s in actual),
                'feature_seconds':seconds,'all_features_xyz_and_metadata_exact':True}
        records.append(record);print(json.dumps(record),flush=True)
        (out/(name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
        del expected,actual,xyz;gc.collect()
    files=['fast_mixed_sequences.py','indexed_surface_features.py','bounded_sequences.py','mixed_sequences.py','surface_features.py','audit_fast_mixed_features.py']
    (out/'receipt.json').write_text(json.dumps({'status':'PASS','groups':records,
        'sources':{str(Path('weld_pipeline')/f):sha256(Path('weld_pipeline')/f) for f in files},
        'scope':'Five real pieces, exact coordinates, all84 features and curve metadata; concurrent-development timings, not formal speed acceptance.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
