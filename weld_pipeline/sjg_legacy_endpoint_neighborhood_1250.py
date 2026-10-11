"""Add frozen pure-input proposal endpoint projections within 3 mm.

No endpoint is obtained from labels; nearby carriers retain alternative
projections instead of treating the prior 0.1 mm routing radius as exact truth.
"""
import argparse
import gc
import time
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_observed_curve_boundaries_1243 import unique_parameters
from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions


def legacy_endpoints(key,legacy_run='run379'):
    folder=OUT/f'{legacy_run}/test_{key}';receipt=read(folder/'receipt.json')
    assert receipt['GT_reads']==receipt['STEP_reads']==0
    endpoints=[]
    if 'reused_artifact_sha256' in receipt:
        for name in ('compact_parents.json','parent_points.npy'):
            assert digest(folder/name)==receipt['reused_artifact_sha256'][name]
        points=np.load(folder/'parent_points.npy',mmap_mode='r')
        for parent in read(folder/'compact_parents.json')['parents']:
            for child in parent['segments']:
                a,b=child['point_range'];endpoints.extend(points[[a,b-1]])
        points._mmap.close()
    else:
        assert digest(folder/'decoded.json')==receipt['decoded_sha256']
        for parent in read_shared_predictions(folder/'decoded.json')['parents']:
            for child in parent['segments']:endpoints.extend(np.asarray(child['points'])[[0,-1]])
    return np.asarray(endpoints,float),digest(folder/'receipt.json')


def prepare(key,dest,base_run='run395',legacy_run='run379'):
    start=time.perf_counter();basefolder=OUT/f'{base_run}/source_{key}';prior=read(basefolder/'receipt.json')
    assert prior['GT_reads']==0 and digest(basefolder/'boundary_events.npz')==prior['events_sha256']
    with np.load(basefolder/'boundary_events.npz') as z:base={k:z[k] for k in z.files}
    endpoints,legacy_receipt_sha256=legacy_endpoints(key,legacy_run)
    case=load_case(key);g=case['graph'];case['x']._mmap.close()
    co=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/co['input_file'])==co['input_sha256']
    raw=np.load(OUT/co['input_file'],mmap_mode='r');tree=cKDTree(raw)
    fields=('q_mm','surface_ids','kinds','hull_signed_excess_mm','nearest_observed_point_mm',
        'plane_residual_max_mm','incidence','finite_support')
    chunks={k:[] for k in fields};offsets=[0];added=0
    for j,pid in enumerate(base['profile_ids']):
        curve=g.curve(int(pid));q,d=curve.project(endpoints)
        extra=unique_parameters(q[d<=3.])
        assert max(float(np.max(-extra,initial=0)),float(np.max(extra-curve.length,initial=0)))<=1e-8
        extra=np.clip(extra,0,curve.length);n=len(extra);added+=n
        new=dict(q_mm=extra,surface_ids=np.full(n,-1),kinds=np.full(n,6),
            hull_signed_excess_mm=np.zeros(n),nearest_observed_point_mm=tree.query(curve.at(extra),workers=2)[0] if n else np.zeros(0),
            plane_residual_max_mm=np.zeros(n),incidence=np.zeros(n),finite_support=np.zeros(n,bool))
        a,b=base['offsets'][j:j+2];order=np.argsort(np.r_[base['q_mm'][a:b],extra],kind='stable')
        for field in fields:chunks[field].append(np.r_[base[field][a:b],new[field]][order])
        offsets.append(offsets[-1]+int(b-a)+n)
    data={k:np.concatenate(v) for k,v in chunks.items()}
    for name,dtype in [('surface_ids',np.int32),('kinds',np.uint8),('finite_support',bool)]:data[name]=data[name].astype(dtype)
    data.update(profile_ids=base['profile_ids'],admitted_rows=base['admitted_rows'],offsets=np.asarray(offsets))
    folder=dest/f'source_{key}';folder.mkdir(exist_ok=False);np.savez(folder/'boundary_events.npz',**data)
    write(folder/'receipt.json',dict(status='INPUT_BOUNDARY_EVENTS_FROZEN',source=key,
        events_sha256=digest(folder/'boundary_events.npz'),prior_events_sha256=prior['events_sha256'],
        input_sha256=co['input_sha256'],legacy_receipt_sha256=legacy_receipt_sha256,
        GT_reads=0,STEP_reads=0,fits=0,projection_radius_mm=3.,events=len(data['q_mm']),
        added_projection_events=added,all_prior_events_retained=True,
        kind6_is_model_proposal_not_observed_boundary=True,seconds=time.perf_counter()-start))
    raw._mmap.close();print('ENDPOINT_NEIGHBORHOOD_READY',key,'ADDED',added,'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True)
    a=ap.parse_args();d=OUT/a.run;d.mkdir(exist_ok=False)
    write(d/'registered_protocol.json',dict(source_sha256=digest(__file__),sources=a.sources,projection_radius_mm=3.,
        GT_reads=0,STEP_reads=0,fit_calls=0,keep_all_prior_events=True,preserve_best379=True))
    for key in a.sources:prepare(key,d);gc.collect()


if __name__=='__main__':main()
