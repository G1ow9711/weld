"""Interval and oriented continuation features from frozen point predictions."""
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,digest
from weld_pipeline.sjg_prepare_776 import OUT
from weld_pipeline.sjg_joint_model_817 import JointEventModel
from weld_pipeline.sjg_pure_data_799 import support_distance


def joint_context(source):
    file=OUT/f'run35/source_{source:02d}_inputs.npz'
    binding=next(r for r in read(OUT/'run35/all_inputs_frozen.json')['sources'] if r['source']==source)
    assert digest(file)==binding['sha256']
    with np.load(file) as z:graph={k:torch.from_numpy(z[k]) for k in z.files}
    model=JointEventModel()
    with (OUT/'run36/last.pt').open('rb') as h:state=torch.load(h,map_location='cpu')
    model.load_state_dict(state['model']);del state;model.eval()
    with torch.inference_mode():
        h=model.encode_faces(graph['face_x'],graph['neighbors'],graph['relation']);chunks=[]
        for start in range(0,len(graph['pairs']),4096):
            chunks.append(model.encode_joints(h,graph['pairs'][start:start+4096],graph['geometry'][start:start+4096]).numpy())
    profiles=read(OUT/f'run34/source_{source:02d}/profiles.json')
    return profiles,np.concatenate(chunks),graph['geometry'].numpy()


def interval_inputs(source,base):
    manifest=read(base/'all_predictions_frozen.json')['sources']
    binding=next(r for r in manifest if r['source']==source)
    assert digest(base/binding['file'])==binding['sha256']
    predictions=read(base/binding['file'])['predictions']
    profiles,h,geometry=joint_context(source)
    fields={r['profile']:r for r in read(base/f'source_{source:02d}_fields.json')['files']}
    cache={};nodes=[]
    for child in predictions:
        ni=child['profile'];p=profiles[ni]
        if ni not in cache:
            field=fields[ni];assert digest(base/field['file'])==field['sha256']
            with np.load(base/field['file']) as z:cache[ni]=(z['q_mm'],z['probability'])
        q,probability=cache[ni];lo,hi=child['lo'],child['hi'];length=max(hi-lo,1e-8);span=max(p['length'],1e-8)
        inside=(q>=lo)&(q<=hi)
        if not inside.any():inside[np.abs(q-(lo+hi)/2).argmin()]=True
        foreground=probability[inside,0]
        support=sum(max(0,min(hi,b)-max(lo,a)) for a,b in p['support_intervals_mm'])/length
        ends=support_distance(p,np.asarray([lo,hi]))/10
        scalar=np.r_[lo/span,hi/span,length/span,np.log1p(length)/10,lo/1000,hi/1000,
            foreground.mean(),foreground.std(),foreground.min(),np.quantile(foreground,.25),np.quantile(foreground,.75),
            child.get('start_event_confidence',0.),child.get('end_event_confidence',0.),child['candidate_probability'],
            support,1-support,ends,float(child['weld_type']=='vertical'),1.]
        assert len(scalar)==20
        nodes.append(np.r_[h[ni],geometry[ni],scalar])
    x=np.asarray(nodes,np.float32).reshape(-1,132)
    assert np.isfinite(x).all()
    return predictions,profiles,x


def continuation_inputs(children,profiles):
    """Both traversal orientations; no STEP/material-ID eligibility filter."""
    points=np.asarray([c['points'] for c in children],float).reshape(-1,2,3)
    n=len(points);length=np.linalg.norm(points[:,1]-points[:,0],axis=1)
    direction=(points[:,1]-points[:,0])/np.maximum(length[:,None],1e-8)
    center=points.mean(1);profile_ids=np.asarray([c['profile'] for c in children])
    face_pairs=np.asarray([profiles[int(i)]['face_indices'] for i in profile_ids],np.int64).reshape(-1,2)
    kinds=np.asarray([c['weld_type']=='vertical' for c in children])
    pair_parts=[];feature_parts=[]
    for first_reverse in (0,1):
        for second_reverse in (0,1):
            da=direction*(1-2*first_reverse);db=direction*(1-2*second_reverse)
            for i in range(n):
                delta=points[:,second_reverse]-points[i,1-first_reverse]
                gap=np.linalg.norm(delta,axis=1);along=delta@da[i]
                perp=np.linalg.norm(delta-along[:,None]*da[i],axis=1)
                tangent=db@da[i];ahead=(center-center[i])@da[i]
                collinear=(tangent>.999)&(along>=-3.)&(perp<=3.)&(ahead>1e-6)
                # Six millimetres is the combined endpoint error allowance;
                # longer collinear gaps remain candidates without a length cap.
                legal=(gap<=6.)|collinear;legal[i]=False;js=np.flatnonzero(legal)
                if not len(js):continue
                shared=np.asarray([len(set(face_pairs[i])&set(face_pairs[j])) for j in js])/2
                geom=np.c_[delta[js]/1000,gap[js]/1000,along[js]/1000,perp[js]/3,tangent[js],
                    np.full(len(js),np.log1p(length[i])/10),np.log1p(length[js])/10,
                    profile_ids[js]==profile_ids[i],shared,kinds[js]==kinds[i],
                    np.full(len(js),first_reverse),np.full(len(js),second_reverse),
                    np.tile(da[i],(len(js),1)),db[js]]
                pair_parts.append(np.c_[np.full(len(js),i),np.full(len(js),first_reverse),js,np.full(len(js),second_reverse)])
                feature_parts.append(geom)
    pairs=np.concatenate(pair_parts).astype(np.int64) if pair_parts else np.empty((0,4),np.int64)
    relation=np.concatenate(feature_parts).astype(np.float32) if feature_parts else np.empty((0,20),np.float32)
    assert relation.shape==(len(pairs),20) and np.isfinite(relation).all()
    return pairs,relation


def training_targets(children,pairs,target):
    """GT identities are targets only; duplicate-compatible nodes stay positive."""
    points=np.asarray([c['points'] for c in children],float).reshape(-1,2,3)
    kinds=np.asarray([c['weld_type'] for c in children]);matches=[[] for _ in children]
    for pi,parent in enumerate(target['parents']):
        for ci,child in enumerate(parent['children']):
            if child['original_kind']!='line':continue
            endpoints=np.asarray(child['world_representation']['endpoints_mm'])
            direct=np.linalg.norm(points-endpoints,axis=-1).max(1)
            reverse=np.linalg.norm(points[:,::-1]-endpoints,axis=-1).max(1)
            good=np.flatnonzero((np.minimum(direct,reverse)<=3.)&(kinds==child['weld_type']))
            for index in good:matches[int(index)].append((pi,ci,bool(direct[index]<=reverse[index])))
    node=np.asarray([bool(m) for m in matches],np.float32);edge=np.zeros((len(pairs),2),np.float32)
    for k,(i,ori_i,j,ori_j) in enumerate(pairs):
        for pi,ci,same_i in matches[int(i)]:
            for pj,cj,same_j in matches[int(j)]:
                if pi!=pj:continue
                forward_i=bool(same_i)!=bool(ori_i);forward_j=bool(same_j)!=bool(ori_j)
                if forward_i!=forward_j:continue
                parent=target['parents'][pi];count=len(parent['children'])
                adjacent=(cj==ci+1) if forward_i else (cj==ci-1)
                closure=parent['is_closed'] and ((ci==count-1 and cj==0) if forward_i else (ci==0 and cj==count-1))
                if adjacent or closure:
                    edge[k,0]=1
                    if closure or parent['gaps'][min(ci,cj)]=='SHARED':edge[k,1]=1
    return node,edge,matches
