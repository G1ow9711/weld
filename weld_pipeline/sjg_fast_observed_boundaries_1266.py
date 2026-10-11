"""Equivalent finite-tube broad phase for observed boundary preparation.
Derived from frozen1243/1244; final projection and support rules unchanged.
Full output parity is required before runtime adoption.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_observed_curve_boundaries_1243 import support_anchors,unique_parameters,TUBE_MM
from weld_pipeline.sjg_multiscale_endpoint_union_1244 import PRECISE_BAND_MM,LEGACY_PROJECTION_MM
from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
from weld_pipeline.sjg_finite_tube_query_1265 import tube_projection

def prepare_support(key,dest,base_run='run393'):
    start=time.perf_counter();folder0=OUT/f'{base_run}/source_{key}';r=read(folder0/'receipt.json')
    assert r['GT_reads']==0 and digest(folder0/'boundary_events.npz')==r['events_sha256']
    with np.load(folder0/'boundary_events.npz') as z:base={k:z[k] for k in z.files}
    case=load_case(key);g=case['graph'];case['x']._mmap.close()
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==cohort['input_sha256']
    xyz=np.load(OUT/cohort['input_file'],mmap_mode='r')
    raw_tree=cKDTree(xyz)
    fields=('q_mm','surface_ids','kinds','hull_signed_excess_mm','nearest_observed_point_mm',
            'plane_residual_max_mm','incidence','finite_support')
    blocks={k:[] for k in fields};offsets=[0];records=[]
    for j,pid in enumerate(base['profile_ids']):
        curve=g.curve(int(pid))
        v=tube_projection(curve,xyz,raw_tree,TUBE_MM)
        q=v['q_mm'];dist=v['distance_mm'];mask=np.ones(len(q),bool)
        anchors,step=support_anchors(q[mask],curve.length,curve.kind=='arc' and curve.arc.is_closed)
        if len(anchors):
            assert max(float(np.max(-anchors)),float(np.max(anchors-curve.length)))<=1e-8
            anchors=np.clip(anchors,0,curve.length)
        a,b=base['offsets'][j:j+2];n=len(anchors)
        new=dict(q_mm=anchors,surface_ids=np.full(n,-1),kinds=np.full(n,3),
            hull_signed_excess_mm=np.zeros(n),nearest_observed_point_mm=raw_tree.query(curve.at(anchors),workers=2)[0] if n else np.zeros(0),
            plane_residual_max_mm=np.full(n,float(dist[mask].max()) if mask.any() else 0),
            incidence=np.zeros(n),finite_support=np.ones(n,bool))
        order=np.argsort(np.r_[base['q_mm'][a:b],anchors],kind='stable')
        for field in fields:blocks[field].append(np.r_[base[field][a:b],new[field]][order])
        offsets.append(offsets[-1]+b-a+n)
        records.append(dict(profile=int(pid),support_points=int(mask.sum()),median_sampling_step_mm=step,
                            proposed_anchors=n,raw_curve_support_not_material_truth=True))
        if j%128==0:print('OBSERVED_BOUNDARIES',key,j+1,'/',len(base['profile_ids']),flush=True)
    data={k:np.concatenate(v) for k,v in blocks.items()}
    for name,dtype in [('surface_ids',np.int32),('kinds',np.uint8),('finite_support',bool)]:data[name]=data[name].astype(dtype)
    data.update(profile_ids=base['profile_ids'],admitted_rows=base['admitted_rows'],offsets=np.array(offsets))
    folder=dest/f'source_{key}';folder.mkdir(exist_ok=False);np.savez(folder/'boundary_events.npz',**data)
    write(folder/'receipt.json',dict(status='INPUT_BOUNDARY_EVENTS_FROZEN',source=key,
        input_sha256=cohort['input_sha256'],prior_events_sha256=r['events_sha256'],
        events_sha256=digest(folder/'boundary_events.npz'),events=len(data['q_mm']),
        new_observed_support_anchors=int(np.sum(data['kinds']==3)),tube_mm=TUBE_MM,
        GT_reads=0,STEP_reads=0,fits=0,all_prior_events_retained=True,
        kind3_residual_field_is_curve_tube_residual_not_fitted_plane=True,
        gap_does_not_force_weld_split=True,seconds=time.perf_counter()-start))
    write(folder/'support_records.json',dict(records=records))
    xyz._mmap.close();print('OBSERVED_BOUNDARIES_READY',key,'ANCHORS',int(np.sum(data['kinds']==3)),
                              'SECONDS',round(time.perf_counter()-start,2),flush=True)

def prepare_multiscale(key,dest,base_run='run394',legacy_run='run379'):
    start=time.perf_counter();basefolder=OUT/f'{base_run}/source_{key}';r=read(basefolder/'receipt.json')
    assert r['GT_reads']==0 and digest(basefolder/'boundary_events.npz')==r['events_sha256']
    with np.load(basefolder/'boundary_events.npz') as z:base={k:z[k] for k in z.files}
    legacyfolder=OUT/f'{legacy_run}/test_{key}';lr=read(legacyfolder/'receipt.json')
    assert lr['GT_reads']==0 and lr['STEP_reads']==0
    endpoints=[]
    if 'reused_artifact_sha256' in lr:
        for name in ('compact_parents.json','parent_points.npy'):
            assert digest(legacyfolder/name)==lr['reused_artifact_sha256'][name]
        allpoints=np.load(legacyfolder/'parent_points.npy',mmap_mode='r')
        for parent in read(legacyfolder/'compact_parents.json')['parents']:
            for child in parent['segments']:
                lo,hi=child['point_range'];endpoints.extend(allpoints[[lo,hi-1]])
        allpoints._mmap.close()
    else:
        assert digest(legacyfolder/'decoded.json')==lr['decoded_sha256']
        for parent in read_shared_predictions(legacyfolder/'decoded.json')['parents']:
            for child in parent['segments']:endpoints.extend(np.asarray(child['points'])[[0,-1]])
    endpoints=np.asarray(endpoints,float)
    case=load_case(key);g=case['graph'];case['x']._mmap.close()
    co=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/co['input_file'])==co['input_sha256']
    points=np.load(OUT/co['input_file'],mmap_mode='r');tree=cKDTree(points)
    fields=('q_mm','surface_ids','kinds','hull_signed_excess_mm','nearest_observed_point_mm',
            'plane_residual_max_mm','incidence','finite_support')
    blocks={k:[] for k in fields};offsets=[0];audit=[]
    for j,pid in enumerate(base['profile_ids']):
        curve=g.curve(int(pid));v=tube_projection(curve,points,tree,PRECISE_BAND_MM)
        q=v['q_mm'];dist=v['distance_mm'];mask=np.ones(len(q),bool)
        exact,step=support_anchors(q[mask],curve.length,curve.kind=='arc' and curve.arc.is_closed)
        if len(endpoints):
            oldq,oldd=curve.project(endpoints);legacy=unique_parameters(oldq[oldd<=LEGACY_PROJECTION_MM])
        else:legacy=np.array([])
        qq=np.r_[exact,legacy]
        if len(qq):
            assert max(float(np.max(-qq)),float(np.max(qq-curve.length)))<=1e-8
            qq=np.clip(qq,0,curve.length)
        n=len(qq);a,b=base['offsets'][j:j+2]
        new=dict(q_mm=qq,surface_ids=np.full(n,-1),kinds=np.r_[np.full(len(exact),4),np.full(len(legacy),5)],
            hull_signed_excess_mm=np.zeros(n),nearest_observed_point_mm=tree.query(curve.at(qq),workers=2)[0] if n else np.zeros(0),
            plane_residual_max_mm=np.zeros(n),incidence=np.zeros(n),
            finite_support=np.r_[np.ones(len(exact),bool),np.zeros(len(legacy),bool)])
        order=np.argsort(np.r_[base['q_mm'][a:b],qq],kind='stable')
        for name in fields:blocks[name].append(np.r_[base[name][a:b],new[name]][order])
        offsets.append(offsets[-1]+b-a+n)
        audit.append(dict(profile=int(pid),precise_support_points=int(mask.sum()),precise_anchors=len(exact),
                          legacy_anchors=len(legacy),median_sampling_step_mm=step))
        if j%128==0:print('MULTISCALE_BOUNDARY',key,j+1,'/',len(base['profile_ids']),flush=True)
    data={k:np.concatenate(v) for k,v in blocks.items()}
    for name,dtype in [('surface_ids',np.int32),('kinds',np.uint8),('finite_support',bool)]:data[name]=data[name].astype(dtype)
    data.update(profile_ids=base['profile_ids'],admitted_rows=base['admitted_rows'],offsets=np.array(offsets))
    folder=dest/f'source_{key}';folder.mkdir(exist_ok=False);np.savez(folder/'boundary_events.npz',**data)
    write(folder/'receipt.json',dict(status='INPUT_BOUNDARY_EVENTS_FROZEN',source=key,
        events_sha256=digest(folder/'boundary_events.npz'),input_sha256=co['input_sha256'],
        prior_events_sha256=r['events_sha256'],legacy_receipt_sha256=digest(legacyfolder/'receipt.json'),
        legacy_endpoints=len(endpoints),new_precise_anchors=int(np.sum(data['kinds']==4)),
        new_legacy_anchors=int(np.sum(data['kinds']==5)),events=len(data['q_mm']),
        GT_reads=0,STEP_reads=0,fits=0,all_prior_events_retained=True,
        kind5_is_frozen_model_proposal_not_observed_material_boundary=True,
        zero_plane_and_hull_fields_for_kinds4_5_are_unknown=True,
        not_fresh_raw_input_timing_acceptance=True,seconds=time.perf_counter()-start))
    write(folder/'support_records.json',dict(records=audit));points._mmap.close()
    print('MULTISCALE_BOUNDARY_READY',key,'EVENTS',len(data['q_mm']),'SECONDS',round(time.perf_counter()-start,2),flush=True)
