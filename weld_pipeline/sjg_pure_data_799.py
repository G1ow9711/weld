"""Frozen pure-XYZ profile graph inputs, then separately sampled supervision."""
import argparse
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import check_resources,utc

DATA=OUT/'run19';PROFILE=OUT/'run18';SURFACE=OUT/'run17';SEED=2026100799


def input_profiles(si):
    row=next(r for r in read(PROFILE/'all_profiles_frozen.json')['outputs'] if r['source']==si)
    path=PROFILE/row['path'];assert digest(path)==row['sha256']
    return read(path)


def node_inputs(si,profiles):
    binding=next(r for r in read(SURFACE/'feature_freeze.json')['outputs'] if r['source']==si)
    path=SURFACE/binding['path'];assert digest(path)==binding['sha256'];features=np.load(path)
    binding=next(r for r in read(SURFACE/'prediction_freeze.json')['outputs'] if r['source']==si)
    path=SURFACE/binding['path'];assert digest(path)==binding['sha256'];probability=np.load(path)
    folder=OUT/f'run16/source_{si:02d}';freeze=read(folder/'input_freeze.json')
    assert digest(folder/'surfaces.json')==freeze['files']['surfaces.json'];faces=read(folder/'surfaces.json')
    nodes=[];centers=[];directions=[];length=[];pairs=[]
    for p in profiles:
        i,j=p['face_indices'];center=(np.asarray(p['p0'])+p['p1'])*.5
        supported=sum(b-a for a,b in p['support_intervals_mm'])
        gaps=p['unobserved_gaps_mm'];longest=max([b-a for a,b in gaps],default=0.)
        nodes.append(np.r_[features[i],features[j],probability[i],probability[j],p['direction'],
            np.log1p(p['length'])/10,abs(np.asarray(faces[i]['normal'])@faces[j]['normal']),
            (np.asarray(faces[i]['center'])-center)/1000,(np.asarray(faces[j]['center'])-center)/1000,
            supported/max(p['length'],1e-6),longest/max(p['length'],1e-6),
            np.log1p(len(p['support_intervals_mm']))/5,np.log1p(len(p['original_contact_ids']))/5,1.])
        centers.append(center);directions.append(p['direction']);length.append(p['length']);pairs.append((i,j))
    nodes=np.asarray(nodes,np.float32);centers=np.asarray(centers);directions=np.asarray(directions);length=np.asarray(length)
    assert nodes.shape[1]==82 and np.isfinite(nodes).all()
    count=len(nodes);neighborhood=[set() for _ in range(count)]
    if count>1:
        nearest=cKDTree(centers).query(centers,k=min(count,9),workers=2)[1]
        for i,ids in enumerate(nearest):neighborhood[i].update(int(j) for j in ids if j!=i)
    by_face={}
    for i,pair in enumerate(pairs):
        for face in pair:by_face.setdefault(face,[]).append(i)
    for ids in by_face.values():
        if len(ids)<2:continue
        ids=np.asarray(ids,np.int64);neighbors=cKDTree(centers[ids]).query(centers[ids],k=min(5,len(ids)),workers=2)[1]
        for i,local in zip(ids,neighbors):neighborhood[i].update(int(j) for j in ids[local] if j!=i)
    neighbors=np.repeat(np.arange(count,dtype=np.int64)[:,None],16,axis=1)
    valid=np.zeros((count,16),np.float32);relation=np.zeros((count,16,8),np.float32)
    for i,items in enumerate(neighborhood):
        selected=sorted(items,key=lambda j:(float(np.linalg.norm(centers[j]-centers[i])),j))[:16]
        for k,j in enumerate(selected):
            neighbors[i,k]=j;valid[i,k]=1
            relation[i,k]=np.r_[float(bool(set(pairs[i])&set(pairs[j]))),(centers[j]-centers[i])/1000,
                directions[i]@directions[j],np.log((length[j]+1)/(length[i]+1))/5,
                min(nodes[j,64:66]),1.]
    return dict(nodes=nodes,neighbors=neighbors,neighbor_valid=valid,relations=relation)


def assign_supervision(profiles,target,nodes):
    origin=np.asarray([p['p0'] for p in profiles]);direction=np.asarray([p['direction'] for p in profiles])
    length=np.asarray([p['length'] for p in profiles]);assignments=[];missing=[];alternatives=[]
    occupied={}
    for pi,parent in enumerate(target['parents']):
        for ci,child in enumerate(parent['children']):
            if child['original_kind']!='line':missing.append(dict(parent=pi,child=ci,reason='CURVE_NOT_IMPLEMENTED'));continue
            endpoints=np.asarray(child['world_representation']['endpoints_mm'])
            q=np.stack([np.einsum('ij,ij->i',point-origin,direction) for point in endpoints],axis=1)
            nearest=origin[:,None]+np.clip(q,0,length[:,None])[:,:,None]*direction[:,None]
            error=np.linalg.norm(nearest-endpoints[None],axis=2).max(1)
            options=np.flatnonzero(error<=3.)
            if not len(options):missing.append(dict(parent=pi,child=ci,reason='NO_PROFILE_WITHIN_3MM'));continue
            ordered=sorted(options,key=lambda k:(round(float(error[k]),1),-float(np.prod(nodes[k,64:66])),-float(nodes[k,77]),int(k)))
            chosen=None
            for k in ordered:
                lo,hi=sorted(map(float,q[k]))
                # Duplicate target children need independent hypotheses rather
                # than silently collapsing into a single event interval.
                if any(max(abs(lo-a),abs(hi-b))<.01 for a,b in occupied.get(int(k),[])):continue
                chosen=int(k);break
            if chosen is None:missing.append(dict(parent=pi,child=ci,reason='DUPLICATE_INTERVAL_CAPACITY'));continue
            lo,hi=sorted(map(float,q[chosen]));occupied.setdefault(chosen,[]).append((lo,hi))
            assignments.append(dict(parent=pi,child=ci,profile=chosen,lo=lo,hi=hi,
                directed=bool(q[chosen,1]>=q[chosen,0]),original_child_id=child['original_child_id']))
            alternatives.append(dict(parent=pi,child=ci,geometric_options=len(options)))
    labels=np.zeros(len(profiles),np.float32)
    for a in assignments:labels[a['profile']]=1
    return labels,dict(original_G=target['original_children'],assigned_children=len(assignments),
        assignments=assignments,missing=missing,alternatives=alternatives,
        assignment_uses_TRAIN_or_posthoc_labels=True,material_pair_identity_unknown=True)


def support_distance(profile,q):
    intervals=np.asarray(profile['support_intervals_mm'])
    index=np.searchsorted(intervals[:,0],q,side='right')-1
    right=np.clip(index+1,0,len(intervals)-1);left=np.clip(index,0,len(intervals)-1)
    distance=np.minimum(np.maximum(q-intervals[left,1],0),np.maximum(intervals[right,0]-q,0))
    distance[index<0]=intervals[0,0]-q[index<0]
    distance[index>=len(intervals)-1]=np.maximum(q[index>=len(intervals)-1]-intervals[-1,1],0)
    return distance


def query_features(profiles,node_ids,q,xyz,tree):
    positions=np.empty((len(q),3));extra=np.empty((len(q),3))
    for ni in np.unique(node_ids):
        ids=np.flatnonzero(node_ids==ni);p=profiles[int(ni)];qq=q[ids]
        positions[ids]=np.asarray(p['p0'])+qq[:,None]*p['direction']
        distance=support_distance(p,qq)
        extra[ids]=np.c_[distance/50,(distance<=1e-6).astype(float),np.full(len(ids),1.)]
    distance,indices=tree.query(positions,k=min(16,len(xyz)),workers=2)
    patch=np.clip((xyz[indices]-positions[:,None,:])/50,-20,20).astype(np.float32)
    length=np.asarray([profiles[int(i)]['length'] for i in node_ids])
    query=np.c_[q/length,q/1000,(length-q)/1000,distance[:,0]/50,extra].astype(np.float32)
    return patch,query


def training_queries(si,profiles,labels,coverage,xyz):
    rng=np.random.default_rng(SEED+si);examples=[];by_profile={}
    for a in coverage['assignments']:
        ni=a['profile'];by_profile.setdefault(ni,[]).append(a)
        for event in (a['lo'],a['hi']):
            for delta in (-6.,-4.,-2.,0.,2.,4.,6.):examples.append((ni,float(np.clip(event+delta,0,profiles[ni]['length']))))
    if coverage['assignments']:
        for _ in range(2048):
            a=coverage['assignments'][int(rng.integers(len(coverage['assignments'])))];ni=a['profile']
            examples.append((ni,float(np.clip(rng.uniform(a['lo'],a['hi']),0,profiles[ni]['length']))))
    negative=np.flatnonzero(labels==0)
    for _ in range(max(4096,8192-len(examples))):
        ni=int(rng.choice(negative if len(negative) else np.arange(len(profiles))))
        examples.append((ni,float(rng.uniform(0,profiles[ni]['length']))))
    node_ids=np.asarray([a for a,b in examples],np.int64);q=np.asarray([b for a,b in examples])
    targets=np.zeros((len(q),3),np.float32);offsets=np.zeros((len(q),2),np.float32);mask=np.zeros_like(offsets)
    for ni,truth in by_profile.items():
        ids=np.flatnonzero(node_ids==ni);values=q[ids]
        for a in truth:targets[ids,0]=np.maximum(targets[ids,0],((values>=a['lo'])&(values<=a['hi'])).astype(float))
        for head,key in enumerate(('lo','hi')):
            events=np.asarray([a[key] for a in truth]);delta=events[None]-values[:,None]
            nearest=delta[np.arange(len(values)),np.abs(delta).argmin(1)]
            targets[ids,head+1]=np.exp(-.5*(nearest/2)**2)
            offsets[ids,head]=nearest/2;mask[ids,head]=abs(nearest)<=3
    patch,query=query_features(profiles,node_ids,q,xyz,cKDTree(xyz))
    return dict(node_ids=node_ids,q_mm=q,patches=patch,query=query,targets=targets,offsets=offsets,offset_mask=mask)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run19');args=ap.parse_args()
    out=OUT/args.run;out.mkdir(exist_ok=False);started=time.monotonic()
    try:
        inventory=read(OUT/'inventory.json');frozen=[]
        write(out/'registered_protocol.json',dict(stage='PURE_XYZ_PROFILE_GRAPH_AND_TRAINING_DATA',sources=TRAIN+VALID+ARC,
            train=TRAIN,holdout=VALID,arc_pending=ARC,node_features=82,maximum_graph_neighbors=16,
            no_profile_count_cap=True,STEP_reads_during_input_generation=0,weld_GT_reads_during_input_generation=0,
            training_query_sampling_uses_TRAIN_labels=True,seed=SEED,source_sha256=digest(__file__),full_contract_pass=False))
        for si in TRAIN+VALID+ARC:
            print('FREEZE_PURE_PROFILE_GRAPH',si,flush=True);check_resources()
            profiles=input_profiles(si);a=node_inputs(si,profiles);file=out/f'source_{si:02d}_inputs.npz'
            np.savez_compressed(file,**a);frozen.append(dict(source=si,path=file.name,sha256=digest(file),nodes=len(profiles)))
            del a,profiles;gc.collect()
        write(out/'input_freeze.json',dict(outputs=frozen,all_graph_inputs_frozen_before_weld_labels=True))
        rows=[]
        for si in TRAIN+VALID+ARC:
            profiles=input_profiles(si)
            with np.load(out/f'source_{si:02d}_inputs.npz',allow_pickle=False) as a:nodes=a['nodes']
            target_path=ROOT/inventory[si]['target']['target_path'];assert digest(target_path)==inventory[si]['target']['target_sha256']
            labels,coverage=assign_supervision(profiles,read(target_path),nodes)
            np.save(out/f'source_{si:02d}_node_labels.npy',labels);write(out/f'source_{si:02d}_supervision.json',coverage)
            queries=0
            if si in TRAIN:
                raw=ROOT/inventory[si]['raw']['original_pcd_path'];assert digest(raw)==inventory[si]['raw']['original_pcd_sha256']
                xyz=read_xyz_pcd(raw)*1000.;data=training_queries(si,profiles,labels,coverage,xyz)
                np.savez_compressed(out/f'source_{si:02d}_train_queries.npz',**data);queries=len(data['node_ids'])
                del xyz,data
            row=dict(source=si,original_G=coverage['original_G'],assigned_children=coverage['assigned_children'],
                     profiles=len(profiles),positive_profiles=int(labels.sum()),training_queries=queries)
            rows.append(row);print('PURE_PROFILE_SUPERVISION',row,flush=True);gc.collect();check_resources()
        write(out/'receipt.json',dict(status='PURE_PROFILE_INPUTS_AND_SUPERVISION_COMPLETE',utc=utc(),rows=rows,
            optimizer_updates=0,full_contract_pass=False,elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write(out/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),optimizer_updates=0));raise


if __name__=='__main__':main()
