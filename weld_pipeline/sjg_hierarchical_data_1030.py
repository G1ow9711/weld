"""Freeze pure-input hierarchy features, then prepare seven TRAIN populations."""
import argparse,gc,shutil,time,traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_surface_material_joint_model_985 import SurfaceMaterialJointModel
from weld_pipeline.sjg_local_event_train_930 import as_tensors
from weld_pipeline.sjg_hierarchical_features_1029 import static_features,IntervalFeatures,link_index


def family_supervision(key,domain,profile_family,transforms,target):
    sup=read(OUT/f'run186/source_{key}_supervision.json')
    original={c['original_child_id']:(p,c) for p in target['parents'] for c in p['children']}
    result={}
    for profile,events in sup['by_profile'].items():
        p=int(profile);fid=int(profile_family[p]);start,sign,length=transforms[p]
        children=result.setdefault(fid,{})
        for event in events:
            child=event['original_child_id'];error=event['maximum_bidirectional_sample_error_mm']
            if child in children and children[child]['error']<=error:continue
            parent,native=original[child];pieces=[];circ=domain['families'][fid]['circumference_mm']
            for a,b in event['intervals']:
                lo,hi=sorted((float(start+sign*a),float(start+sign*b)))
                if circ:
                    if hi-lo>=circ-1e-8:pieces.append((0.,circ));continue
                    span=hi-lo;lo%=circ;hi=lo+span
                    if hi>circ:pieces.extend([(lo,circ),(0.,hi-circ)]);continue
                pieces.append((lo,hi))
            kind=native.get('weld_type');types={'horizontal':0,'vertical':1}
            children[child]=dict(error=error,pieces=pieces,child=child,parent=event['parent'],
                segment=event['child'],forward=bool(event['target_direction_forward'])==(sign>0),
                weld_type=types.get(kind,-1),closed=bool(event['closed_whole']))
    return result,sup


def make_nodes(domain,supervision):
    rows=[];labels=[];owners=[]
    for fid,family in enumerate(domain['families']):
        segments=[(a,b,meta) for meta in supervision.get(fid,{}).values() for a,b in meta['pieces']]
        stack=[(a,b,0) for a,b in reversed(family['domains_mm'])]
        while stack:
            lo,hi,depth=stack.pop();active=[s for s in segments if min(hi,s[1])-max(lo,s[0])>1e-9]
            owner=None;bounds=[0.,1.];state=0
            if active:
                if len(active)==1 and active[0][0]<=lo and active[0][1]>=hi:
                    state=1;owner=active[0][2]
                elif hi-lo<=8 and len(active)==1:
                    state=2;s=active[0];bounds=[(max(lo,s[0])-lo)/(hi-lo),(min(hi,s[1])-lo)/(hi-lo)];owner=s[2]
                else:
                    if depth>=48:raise RuntimeError('Unresolved multi-child interval; refuse to erase labels')
                    state=3;mid=(lo+hi)/2;stack.extend([(mid,hi,depth+1),(lo,mid,depth+1)])
            rows.append((fid,lo,hi));labels.append((state,*bounds));owners.append(owner)
    x=np.asarray(rows,np.float64);y=np.asarray(labels,np.float32)
    return x,y,owners


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);(dest/'source_snapshot').mkdir()
    for name in ('sjg_hierarchical_features_1029.py','sjg_hierarchical_data_1030.py','sjg_hierarchical_joint_model_1028.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,dest/'source_snapshot'/name)
    torch.set_num_threads(2)
    protocol=dict(stage='HIERARCHICAL_TRAIN_POPULATION',input_domain='run201',point_features='run166',
        structure_encoder='run187/final.pt',TRAIN_G=363,all_G=500,all_static_inputs_frozen_before_labels=True,
        labels='geometric compatibility, not customer material identity',sampling='full enumerated teacher-forced population; no positive reweight',
        inference_must_predict_all_tree_states=True,source_identity_not_model_input=True,STEP_not_runtime_input=True,
        optimizer_updates=0,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        checkpoint=OUT/'run187/final.pt';frozen_sha=digest(checkpoint)
        encoder=SurfaceMaterialJointModel();encoder.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict']);encoder.eval()
        cases=read(OUT/'run98/local_cohort.json')['cases'];bindings=[]
        oldgraphs={r['source_key']:r for r in read(OUT/'run186/all_graphs_frozen_before_labels.json')['cases']}
        domains={r['source_key']:r for r in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases']}
        for case in cases:
            check_resources();key=case['source_key'];start=time.perf_counter()
            assert digest(OUT/'run186'/oldgraphs[key]['file'])==oldgraphs[key]['sha256']
            assert digest(OUT/'run201'/domains[key]['json_file'])==domains[key]['json_sha256']
            tensors=as_tensors(OUT/'run186'/oldgraphs[key]['file'])
            with torch.inference_mode():latent=encoder.encode_structure(tensors)[0].numpy()
            graph={k:v.numpy() for k,v in tensors.items()};domain=read(OUT/'run201'/domains[key]['json_file'])
            features=static_features(domain,graph,latent);path=dest/f'source_{key}_static.npz';np.savez(path,**features)
            bindings.append(dict(source_key=key,file=path.name,sha256=digest(path),relations=len(features['relation_x']),
                families=len(features['family_x']),relation_features=features['relation_x'].shape[1],family_features=features['family_x'].shape[1],
                domain_sha256=domains[key]['json_sha256'],XYZ_sha256=case['input_sha256']))
            print('HIERARCHY_INPUT_FROZEN',key,len(features['relation_x']),len(features['family_x']),round(time.perf_counter()-start,2),flush=True)
            del domain,tensors,graph,features,latent;gc.collect()
        write(dest/'all_static_inputs_frozen_before_labels.json',dict(cases=bindings,encoder_sha256=frozen_sha,target_reads=0))
        train_keys=[r['source_key'] for r in read(OUT/'run186/training_manifest.json')['cases']];manifest=[]
        del encoder;gc.collect()
        for case in cases:
            key=case['source_key']
            if key not in train_keys:continue
            check_resources();start=time.perf_counter();domain=read(OUT/f'run201/source_{key}_joint_domain.json')
            with np.load(OUT/f'run201/source_{key}_joint_mapping.npz',allow_pickle=False) as z:
                mapping=z['profile_family'];transforms=z['local_to_canonical']
            assert digest(OUT/case['target_file'])==case['target_sha256'];target=read(OUT/case['target_file'])
            supervised,sup=family_supervision(key,domain,mapping,transforms,target)
            nodes,label,owners=make_nodes(domain,supervised)
            print('HIERARCHY_TRAIN_NODES',key,len(nodes),'REFERENCE_G',sup['original_G'],flush=True)
            xyz=np.load(OUT/case['input_file'],mmap_mode='r',allow_pickle=False)
            fb=read(OUT/f'run166/source_{key}_freeze.json');assert fb['input_sha256']==case['input_sha256']
            feature_path=OUT/'run166'/fb['feature_file'];assert digest(feature_path)==fb['feature_sha256']
            with np.load(feature_path,allow_pickle=False) as z:point={k:z[k] for k in ('embedding','normal','boundary')}
            profiles=read(OUT/f'run185/source_{key}/profiles.json')
            provider=IntervalFeatures(domain,profiles,transforms,xyz,point)
            x=provider.batch(nodes[:,0].astype(np.int64),nodes[:,1],nodes[:,2])
            family_y=np.zeros(len(domain['families']),np.float32);relation_y=np.zeros(len(domain['relations']),np.float32)
            for fid in supervised:family_y[fid]=1;relation_y[domain['families'][fid]['relation']]=1
            known=np.asarray([o is not None for o in owners]);direction=np.asarray([float(o['forward']) if o else 0 for o in owners],np.float32)
            types=np.asarray([o['weld_type'] if o else -1 for o in owners],np.int64)
            leaves=np.flatnonzero(known)
            edges=link_index(nodes[leaves,0].astype(np.int64),nodes[leaves,1],nodes[leaves,2],domain)
            native_children={c['original_child_id']:c for p in target['parents'] for c in p['children']}
            child_codes={child:i for i,child in enumerate(sorted(native_children))}
            owner_rows=[owners[i] for i in leaves]
            edge_labels=dict(leaf_nodes=leaves,
                leaf_child=np.asarray([child_codes[o['child']] for o in owner_rows],np.int64),
                leaf_parent=np.asarray([o['parent'] for o in owner_rows],np.int64),
                leaf_segment=np.asarray([o['segment'] for o in owner_rows],np.int64),
                leaf_true_begin=np.asarray([native_children[o['child']]['original_points_mm'][0] for o in owner_rows],np.float64),
                leaf_true_end=np.asarray([native_children[o['child']]['original_points_mm'][-1] for o in owner_rows],np.float64))
            path=dest/f'source_{key}_train.npz'
            np.savez(path,relation_y=relation_y,family_y=family_y,node_family=nodes[:,0].astype(np.int64),
                node_lo=nodes[:,1],node_hi=nodes[:,2],interval_x=x,state=label[:,0].astype(np.int64),bounds=label[:,1:],
                bounds_mask=label[:,0]==2,direction=direction,direction_mask=known,weld_type=types,weld_type_mask=types>=0,
                **edges,**edge_labels)
            row=dict(source_key=key,G=case['original_G'],file=path.name,sha256=digest(path),nodes=len(nodes),
                state_counts=np.bincount(label[:,0].astype(int),minlength=4).tolist(),relation_positive=int(relation_y.sum()),
                relation_total=len(relation_y),family_positive=int(family_y.sum()),family_total=len(family_y),
                links=int(edges['link_offsets'][-1]),link_training='uniform resampled complete CSR domain; q=1/N',
                input_queries=provider.query_count,preparation_seconds=time.perf_counter()-start)
            manifest.append(row);write(dest/'progress.json',dict(TRAIN_prepared=manifest,utc=utc()))
            print('HIERARCHY_TRAIN_READY',key,'nodes',len(nodes),'queries',provider.query_count,'seconds',round(row['preparation_seconds'],2),flush=True)
            del domain,nodes,label,owners,xyz,point,profiles,provider,x,edges,edge_labels,supervised;gc.collect()
        assert sum(r['G'] for r in manifest)==363 and digest(checkpoint)==frozen_sha
        write(dest/'training_manifest.json',dict(cases=manifest,TRAIN_G=363,encoder_sha256=frozen_sha,static_bindings=bindings))
        write(dest/'receipt.json',dict(status='HIERARCHICAL_POPULATION_READY',TRAIN_G=363,all_G=500,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),utc=utc()));raise


if __name__=='__main__':main()
