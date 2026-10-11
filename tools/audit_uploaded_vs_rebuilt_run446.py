"""Compare available original and raw-XYZ rebuilt stages; no selection changes."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest
BASE=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'

def main():
    result=[]
    for key in ('21','BM79A_native_local','10','25','36'):
        rebuilt=BASE/f'source_{key}';original_baseline='run413' if key in ('25','36') else 'run379';records=[]
        stages=[('baseline',OUT/original_baseline/f'test_{key}',rebuilt/'run379'/f'test_{key}'),('proposals',OUT/'run432'/f'proposal_{key}',rebuilt/'run432'/f'proposal_{key}'),('parent_features',OUT/'run433'/f'source_{key}',rebuilt/'run433'/f'source_{key}'),('child_features',OUT/'run434'/f'source_{key}',rebuilt/'run434'/f'source_{key}'),('final',OUT/'run446'/f'test_{key}',rebuilt/'run446'/f'test_{key}')]
        for stage,a,b in stages:
            row=dict(stage=stage,original=str(a),rebuilt=str(b),files=[])
            for name in ('final_decoded.json','decoded.json','features.npy','metadata.json','origin.json','selected_graph.json','decisions.json'):
                x=a/name;y=b/name
                if not x.exists() or not y.exists():continue
                v=dict(name=name,original_sha256=digest(x),rebuilt_sha256=digest(y));v['bytes_equal']=v['original_sha256']==v['rebuilt_sha256']
                if name=='features.npy':
                    u=np.load(x,mmap_mode='r');w=np.load(y,mmap_mode='r');v.update(original_shape=list(u.shape),rebuilt_shape=list(w.shape))
                    if u.shape==w.shape:v.update(same_row_order_max_absolute_difference=float(np.max(np.abs(u-w))) if u.size else 0.)
                if name in ('final_decoded.json','decoded.json'):
                    for label,path in [('original',x),('rebuilt',y)]:
                        d=read(path);v[label+'_parents']=len(d.get('parents',[]));v[label+'_declared_children']=d.get('total_prediction_children')
                row['files'].append(v)
            records.append(row)
        # Evaluate score differences from already-completed authoritative scoring outputs.
        original=read(ROOT/f'server_rebuild/run446_uploaded_input_inference_v1/test_{key}/evaluation.json')['report']
        rebuilt_score=read(rebuilt/f'run446/test_{key}/evaluation.json')['categories']['overall']
        result.append(dict(key=key,final_original=original,final_rebuilt=rebuilt_score,stages=records))
    dest=ROOT/'server_rebuild/run446_uploaded_vs_rebuilt_audit_v1';dest.mkdir(exist_ok=False)
    write(dest/'report.json',dict(cases=result,scope='Available stage hash/shape and final metrics. Raw row differences do not prove semantic differences without candidate alignment. Missing original upstream stages are not reconstructed here.',script_sha256=digest(__file__),protected_DEV_reads=0))
    for r in result:
        print(r['key'],'original TP',r['final_original']['TP'],'rebuilt TP',r['final_rebuilt']['TP'],[(s['stage'],[(v['name'],v.get('original_shape'),v.get('rebuilt_shape')) for v in s['files'] if v['name']=='features.npy']) for s in r['stages']],flush=True)

if __name__=='__main__':main()
