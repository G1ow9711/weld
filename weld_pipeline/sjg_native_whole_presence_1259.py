"""Audit whole-carrier presence supervision before another classifier fit.

Original native GT defines supervision only. Geometry and local point features
are frozen before label access. Candidate positions, geometry and denominators
are unchanged. The two face-order rows share existence of the same geometric
carrier; this does not certify material roles or directed weld attributes.
"""
import gc
import shutil
import time
import traceback
from collections import defaultdict
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_recall_set_worker_1162 import target_for
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.sjg_native_carrier_compatibility_1252 import arc_matches,controls

TRAIN=('21','BM79A_native_local','10','25','36')


def paths(key):
    return ('run414/whole','run414/local') if key in ('25','36') else ('run380','run383')


def complete_background(case,key):
    graph=case['graph'];canonical=np.asarray(case['canonical'])
    pids=np.searchsorted(graph.offsets,canonical,side='right')-1
    lp=OUT/f'run318/source_{key}_count_targets.npz';ip=OUT/f'run318/source_{key}_interval_instances.json'
    binding=next(v for v in read(OUT/'run342/receipt.json')['cases'] if v['source']==key)
    assert digest(lp)==binding['labels_sha256'] and digest(ip)==binding['instances_sha256']
    with np.load(lp) as z:old=z['labels']
    known=np.zeros(2*len(graph.profiles),int);required=np.repeat(np.diff(graph.offsets),2)
    for side in (0,1):np.add.at(known,2*pids+side,old[side::2]>=0)
    complete=(known==required)&(required>0)
    cats=defaultdict(set)
    for e in read(ip)['instances']:cats[2*int(pids[e['row']//2])+e['row']%2,tuple(e['owner'])].add(e['label'])
    conflicts=[]
    for (row,owner),values in cats.items():
        if len(values)>1:complete[row]=False;conflicts.append(dict(row=row,owner=list(owner),categories=sorted(values)))
    return complete,dict(labels_sha256=binding['labels_sha256'],instances_sha256=binding['instances_sha256'],conflicts=conflicts)


def line_parameters(curve,endpoints):
    """Intervals on a finite line inside each endpoint's unchanged 3 mm ball."""
    origin=curve.at(np.array([0.]))[0];end=curve.at(np.array([curve.length]))[0]
    unit=(end-origin)/curve.length;delta=endpoints-origin
    centers=delta@unit;perp=np.linalg.norm(delta-centers[:,None]*unit,axis=1)
    if max(perp)>3.:return None
    width=np.sqrt(np.maximum(0,9-perp**2));lo=np.maximum(0,centers-width);hi=np.minimum(curve.length,centers+width)
    if np.any(lo>hi):return None
    q=np.clip(centers,lo,hi)
    if abs(q[1]-q[0])<1e-9:
        options=[np.array([lo[0],hi[1]]),np.array([hi[0],lo[1]])]
        q=max(options,key=lambda v:abs(v[1]-v[0]))
    return q if abs(q[1]-q[0])>1e-9 else None


def prepare(key,dest):
    begin=time.perf_counter();check_resources();case=load_case(key);g=case['graph'];case['x']._mmap.close()
    original,bind=target_for(key);target=native_targets(original)
    truths=[];refs=[];owners=[];closed=[]
    for pi,p in enumerate(target['parents']):
        for ci,c in enumerate(p['children']):
            truths.append(c);refs.append(reference_segment(c));owners.append([pi,ci])
            closed.append(bool(p['is_closed'] and len(p['children'])==1))
    ends=np.asarray([np.asarray(v['points'])[[0,-1]] for v in refs])
    labels,bg=complete_background(case,key);labels=np.where(labels,0,-1).astype(np.int8)
    profiles=[];offsets=[0];identities=[];covered=set();examples=[];uncertain=[]
    kinds=np.array([v['original_kind'] for v in truths]);types=[v['weld_type'] for v in truths]
    for pid,p in enumerate(g.profiles):
        curve=g.curve(pid);q,dd=curve.project(ends.reshape(-1,3));dd=dd.reshape(-1,2);q=q.reshape(-1,2)
        possible=np.flatnonzero((kinds==curve.kind)&((dd.max(1)<=3.+1e-8)|np.asarray(closed)))
        found=[]
        for oi in possible:
            oi=int(oi)
            if curve.kind=='line':
                parameters=line_parameters(curve,ends[oi])
                if parameters is None:
                    if dd[oi].max()<=3.+1e-8:uncertain.append((pid,oi))
                    continue
                projected=curve.at(parameters)
                endpoint_error=float(np.linalg.norm(projected-ends[oi],axis=1).max())
                # Linear interpolation between corresponding endpoints bounds
                # both complete straight curves, exactly as the audited1252 rule.
                accepted=endpoint_error<=3.
                if accepted and (endpoint_error>=3.-1e-8 or len(examples)<80 or pid%257==0):
                    predicted=dict(line_segment(*projected),weld_type=types[oi])
                    accepted=child_error(predicted,truths[oi],refs[oi],closed[oi],False) is not None
                if not accepted:uncertain.append((pid,oi))
            else:
                values=np.unique(np.r_[0.,curve.length,q[oi]])
                pairs=[i*len(values)+j for i in range(len(values)) for j in range(i+1,len(values))]
                accepted=bool(arc_matches(curve,values,pairs,truths[oi],refs[oi],closed[oi]))
                parameters=None
            if accepted:
                found.append(oi);covered.add(oi)
                if len(examples)<80:examples.append(dict(profile=pid,owner_index=oi,parameters_mm=None if parameters is None else parameters.tolist()))
        if found:
            profiles.append(pid);identities.extend(found);offsets.append(len(identities));labels[2*pid:2*pid+2]=1
        if (pid+1)%16384==0:print('NATIVE_PRESENCE',key,pid+1,'/',len(g.profiles),flush=True)
    for pid,_ in uncertain:
        for row in (2*pid,2*pid+1):
            if labels[row]==0:labels[row]=-1
    old=None;whole,_=paths(key);old_path=OUT/f'{whole}/source_{key}/joint_count_targets.npz'
    if old_path.exists():
        with np.load(old_path) as z:old=z['labels']
    folder=dest/f'source_{key}';folder.mkdir()
    np.savez(folder/'presence_targets.npz',labels=labels,positive_profiles=np.asarray(profiles,np.int64),
        offsets=np.asarray(offsets,np.int64),owner_indices=np.asarray(identities,np.int64))
    write(folder/'owner_index.json',dict(owners=owners,weld_types=types,examples=examples,
        original_target_sha256=bind['target_sha256'],uncertain_projection_relations=uncertain))
    summary=dict(source=key,G=len(truths),covered_G=len(covered),rows=len(labels),positive_rows=int((labels==1).sum()),
        negative_rows=int((labels==0).sum()),unknown_rows=int((labels<0).sum()),
        native_positive_previously_known_negative=None if old is None else int(((labels==1)&(old==0)).sum()),
        old_positive_failing_native=None if old is None else int(((old>0)&(labels!=1)).sum()),
        positive_original_children_by_type={t:sum(i in covered and types[i]==t for i in range(len(types))) for t in set(types)},
        missing_original_children=[dict(parent=owners[i][0],child=owners[i][1],weld_type=types[i]) for i in range(len(owners)) if i not in covered],
        target_sha256=bind['target_sha256'],background_binding=bg,labels_sha256=digest(folder/'presence_targets.npz'),
        geometrically_positive_both_face_order_rows=True,material_role_labels_not_fabricated=True,
        GT_geometry_used_only_to_define_supervision=True,GT_endpoint_insertion_into_inputs=False,
        fit_calls=0,DEV_reads=0,full_contract_pass=False,seconds=time.perf_counter()-begin)
    write(folder/'receipt.json',summary);print('NATIVE_PRESENCE_READY',summary,flush=True)
    return summary


def main():
    dest=OUT/'run419';dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir()
    for name in ('sjg_native_whole_presence_1259.py','sjg_native_carrier_compatibility_1252.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snapshot/name)
    bindings={}
    for key in TRAIN:
        whole,local=paths(key);w=OUT/f'{whole}/source_{key}';l=OUT/f'{local}/source_{key}'
        wr=read(w/'receipt.json');lr=read(l/'receipt.json')
        assert wr['GT_reads']==lr['GT_reads']==0
        assert digest(w/'joint_features.npy')==wr['feature_sha256'] and digest(l/'local_features.npy')==lr['feature_sha256']
        bindings[key]=dict(whole_run=whole,local_run=local,whole_features_sha256=wr['feature_sha256'],local_features_sha256=lr['feature_sha256'])
    write(dest/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=[],G=269,bindings=bindings,
        input_features_frozen_before_this_supervision=True,source_sha256=digest(__file__),fit_calls=0,
        original_targets_unchanged=True,external_consultation_paused=True,full_contract_pass=False))
    try:
        controls(dest);cases=[]
        for key in TRAIN:cases.append(prepare(key,dest));gc.collect()
        assert sum(v['G'] for v in cases)==269
        write(dest/'receipt.json',dict(status='NATIVE_WHOLE_PRESENCE_LABEL_AUDIT_PENDING_PREFIT_CONTROLS',cases=cases,
            G=269,covered_G=sum(v['covered_G'] for v in cases),fit_calls=0,DEV_reads=0,full_contract_pass=False))
    except BaseException:write(dest/'failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise


if __name__=='__main__':main()
