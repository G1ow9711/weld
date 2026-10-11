"""Free-input hierarchical inference. No weld label files are read."""
import argparse,time,traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_shared_joint_domain_1023 import build
from weld_pipeline.sjg_hierarchical_features_1029 import static_features,IntervalFeatures,link_candidates,positions
from weld_pipeline.sjg_hierarchical_joint_model_1028 import HierarchicalJointModel
from weld_pipeline.sjg_balanced_hierarchy_1043 import load_model
from weld_pipeline.sjg_surface_material_joint_model_985 import SurfaceMaterialJointModel
from weld_pipeline.sjg_local_event_train_930 import as_tensors
from weld_pipeline.contract_curve_geometry import CircularArc


def representation(family,lo,hi,forward=True):
    endpoints=positions(family,np.asarray([lo,hi]))
    if family['kind']=='line':
        return dict(kind='line',endpoints_mm=(endpoints if forward else endpoints[::-1]).tolist())
    d=np.asarray(family['descriptor']);normal=d[3:6];radius=d[6]
    start=endpoints[0] if forward else endpoints[1];sweep=(hi-lo)/radius*(1 if forward else -1)
    closed=abs(abs(sweep)-2*np.pi)<1e-7
    arc=CircularArc(d[:3],radius,normal,(start-d[:3])/radius,sweep,closed)
    return dict(kind='arc',arc_parameters=arc.to_parameters())


def assemble(leaves,link_pairs,link_probability,domain):
    parent=list(range(len(leaves)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    decisions=[]
    for (i,j),prob in zip(link_pairs,link_probability):
        i=int(i);j=int(j)
        if prob[0]<.5:continue
        a,b=leaves[i],leaves[j];fa=domain['families'][a['family']];fb=domain['families'][b['family']]
        ra=domain['relations'][fa['relation']];rb=domain['relations'][fb['relation']]
        if ra['side_hypothesis'] is not None and rb['side_hypothesis'] is not None and ra['side_hypothesis']!=rb['side_hypothesis']:continue
        # Same-child is learned, and numerical equality only permits reuse of
        # a curve coordinate system. No 3 mm trimming or confidence sum.
        if fa['kind']!=fb['kind'] or not np.allclose(fa['descriptor'],fb['descriptor'],atol=1e-8,rtol=0):continue
        if max(a['lo'],b['lo'])>min(a['hi'],b['hi'])+1e-8:continue
        if a['forward']!=b['forward'] or a['weld_type']!=b['weld_type']:continue
        x,y=find(i),find(j)
        if x!=y:parent[y]=x;decisions.append(dict(left=i,right=j,same_child_probability=float(prob[0])))
    groups={}
    for i in range(len(leaves)):groups.setdefault(find(i),[]).append(i)
    predictions=[];leaf_to_prediction={}
    for ids in groups.values():
        exemplar=leaves[max(ids,key=lambda i:leaves[i]['confidence'])];lo=min(leaves[i]['lo'] for i in ids);hi=max(leaves[i]['hi'] for i in ids)
        family=domain['families'][exemplar['family']]
        prediction=dict(representation=representation(family,lo,hi,exemplar['forward']),
            confidence=float(max(leaves[i]['confidence'] for i in ids)),family=exemplar['family'],
            relation=family['relation'],weld_type=exemplar['weld_type'],direction_forward=exemplar['forward'],
            source_leaf_indices=ids,E16_status='UNKNOWN_NOT_ASSESSED',parent_index=None)
        for i in ids:leaf_to_prediction[i]=len(predictions)
        predictions.append(prediction)
    # Directed parent edges are learned independently of same-child merging.
    candidates={}
    for (i,j),prob in zip(link_pairs,link_probability):
        a,b=leaf_to_prediction[int(i)],leaf_to_prediction[int(j)]
        if a==b or prob[1]<.5:continue
        candidates[(a,b)]=max(candidates.get((a,b),0.),float(prob[1]))
    successors={};predecessors={};parent_edges=[]
    for (a,b),score in sorted(candidates.items(),key=lambda x:(-x[1],x[0])):
        if a in successors or b in predecessors:continue
        cursor=b;cycle=False
        while cursor in successors:
            cursor=successors[cursor]
            if cursor==a:cycle=True;break
        if cycle:continue  # Cyclic edge truth remains unavailable, not guessed.
        successors[a]=b;predecessors[b]=a;parent_edges.append(dict(left=a,right=b,probability=score))
    paths=[]
    for begin in range(len(predictions)):
        if begin in predecessors:continue
        path=[begin]
        while path[-1] in successors:path.append(successors[path[-1]])
        index=len(paths)
        for child in path:predictions[child]['parent_index']=index
        paths.append(dict(children=path,closed_parent_status='UNKNOWN',attributes_status='UNKNOWN_NOT_PASSED'))
    return predictions,dict(same_child_decisions=decisions,parent_edges=parent_edges,parents=paths,
        topology_acceptance_not_assessed=True,unknown_closure_not_counted_as_pass=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--training-run',default='run207');ap.add_argument('--checkpoint',default='final.pt');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=True);folder=dest/f'source_{args.source}';folder.mkdir(exist_ok=False)
    start=time.perf_counter();tokens=0;provider=None
    def budget():
        if time.perf_counter()-start>60:raise RuntimeError('Whole workpiece failed: 60 second quick-trial budget exceeded')
        if tokens>100000:raise RuntimeError('Whole workpiece failed: 100000 relation/family/interval tokens exceeded')
    protocol=dict(source=args.source,training_run=args.training_run,target_reads=0,STEP_reads=0,
        cached_inputs=['PTv3 point features','finite surface proposals','surface/material input graph'],
        rebuilt_this_run=['joint domains','hierarchy static features','all neural decisions','adaptive XYZ queries','parent decoding'],
        is_uncached_contract_timing=False,threshold=.5,quick_seconds_limit=60,query_limit=100000,
        full_contract_pass=False,start_utc=utc())
    write(folder/'registered_protocol.json',protocol)
    try:
        check_resources();torch.set_num_threads(2)
        state=torch.load(OUT/args.training_run/args.checkpoint,map_location='cpu');model=load_model(state)
        training_protocol=read(OUT/args.training_run/'registered_protocol.json')
        training_manifest=read(OUT/training_protocol['data_run']/'training_manifest.json')
        assert digest(OUT/'run187/final.pt')==training_manifest['encoder_sha256']
        model.load_state_dict(state['state_dict']);model.eval();device=torch.device('cuda');model.to(device)
        score_semantics=getattr(model,'score_semantics','unweighted_sigmoid_model_score')
        protocol['score_semantics']=score_semantics;write(folder/'registered_protocol.json',protocol)
        case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==args.source)
        base=OUT/f'run185/source_{args.source}';binding=read(base/'input_freeze.json')
        assert digest(base/'profiles.json')==binding['files']['profiles.json']
        assert digest(OUT/case['input_file'])==case['input_sha256']==binding['input_sha256']
        profiles=read(base/'profiles.json');domain,mapping=build(profiles);budget()
        graphs={r['source_key']:r for r in read(OUT/'run186/all_graphs_frozen_before_labels.json')['cases']}
        gb=graphs[args.source];assert digest(OUT/'run186'/gb['file'])==gb['sha256']
        graph=as_tensors(OUT/'run186'/gb['file']);encoder=SurfaceMaterialJointModel()
        encoder.load_state_dict(torch.load(OUT/'run187/final.pt',map_location='cpu')['state_dict']);encoder.eval()
        with torch.inference_mode():face=encoder.encode_structure(graph)[0].numpy()
        static=static_features(domain,{k:v.numpy() for k,v in graph.items()},face)
        with torch.inference_mode():
            r,rl=model.relations(torch.from_numpy(static['relation_x']).to(device));tokens+=len(r);budget()
            rp=rl.sigmoid().cpu().numpy();selected_relations=np.flatnonzero(rp>=.5)
            tested_families=np.flatnonzero(np.isin(static['family_relation'],selected_relations));tokens+=len(tested_families);budget()
            f,fl=model.families(r,torch.from_numpy(static['family_relation'][tested_families]).to(device),
                torch.from_numpy(static['family_x'][tested_families]).to(device))
            fp=np.zeros(len(domain['families']),np.float32);fp[tested_families]=fl.sigmoid().cpu().numpy()
            latent=torch.zeros((len(fp),model.configuration['hidden']),device=device);latent[tested_families]=f
            selected=np.flatnonzero(fp>=.5);queue=[(int(fid),lo,hi,0) for fid in selected for lo,hi in domain['families'][int(fid)]['domains_mm']]
            write(folder/'selection.json',dict(relations=len(rp),selected_relations=selected_relations.tolist(),
                tested_families=len(tested_families),selected_families=selected.tolist(),max_relation_probability=float(rp.max()),
                max_family_probability=float(fp.max()),score_semantics=score_semantics,
                legacy_probability_field_names_do_not_imply_calibration=True,target_reads=0))
            np.save(folder/'relation_probability.npy',rp);np.save(folder/'family_probability.npy',fp)
            leaves=[];leaf_latents=[];invalid_intervals=0
            if queue:
                xyz=np.load(OUT/case['input_file'],mmap_mode='r',allow_pickle=False);fb=read(OUT/f'run166/source_{args.source}_freeze.json')
                assert fb['input_sha256']==case['input_sha256']
                feature_file=OUT/'run166'/fb['feature_file'];assert digest(feature_file)==fb['feature_sha256']
                with np.load(feature_file,allow_pickle=False) as z:point={k:z[k] for k in ('embedding','normal','boundary')}
                provider=IntervalFeatures(domain,profiles,mapping['local_to_canonical'],xyz,point,max_queries=100000)
            while queue:
                budget();tokens+=len(queue);budget();next_queue=[]
                for offset in range(0,len(queue),2048):
                    batch=queue[offset:offset+2048];fid=np.asarray([v[0] for v in batch],np.int64)
                    lo=np.asarray([v[1] for v in batch]);hi=np.asarray([v[2] for v in batch]);x=provider.batch(fid,lo,hi);budget()
                    out=model.intervals(latent,torch.from_numpy(fid).to(device),torch.from_numpy(x).to(device))
                    probs=out['state_logits'].softmax(-1).cpu().numpy();states=probs.argmax(-1);bounds=out['bounds'].cpu().numpy()
                    directions=out['direction_logit'].sigmoid().cpu().numpy();types=out['weld_type_logits'].argmax(-1).cpu().numpy()
                    for i,(family,left,right,depth) in enumerate(batch):
                        cls=int(states[i])
                        if cls==0:continue
                        if cls==3 or (cls==2 and right-left>8):
                            if depth>=48:raise RuntimeError('Unresolved adaptive interval at maximum depth')
                            mid=(left+right)/2
                            if not left<mid<right:raise RuntimeError('Numerically indivisible predicted interval')
                            next_queue.extend([(family,left,mid,depth+1),(family,mid,right,depth+1)]);continue
                        if cls==2:
                            a,b=left+(right-left)*bounds[i];left=float(a);right=float(b)
                            if right-left<=1e-8:invalid_intervals+=1;continue
                        rid=domain['families'][family]['relation'];confidence=float((rp[rid]*fp[family]*probs[i,cls])**(1/3))
                        leaves.append(dict(family=family,lo=left,hi=right,forward=bool(directions[i]>=.5),
                            weld_type=['horizontal','vertical'][int(types[i])],confidence=confidence))
                        leaf_latents.append(out['latent'][i].detach().cpu())
                queue=next_queue
            budget()
            if leaves:
                family=np.asarray([x['family'] for x in leaves]);lo=np.asarray([x['lo'] for x in leaves]);hi=np.asarray([x['hi'] for x in leaves])
                pairs,edge=link_candidates(family,lo,hi,domain,limit=100000);budget()
                hidden=torch.stack(leaf_latents).to(device);probs=[]
                for offset in range(0,len(pairs),4096):
                    sl=slice(offset,offset+4096)
                    probs.append(model.links(hidden,torch.from_numpy(pairs[sl]).to(device),torch.from_numpy(edge[sl]).to(device)).sigmoid().cpu().numpy())
                probability=np.concatenate(probs) if probs else np.empty((0,5))
                predictions,parent_graph=assemble(leaves,pairs,probability,domain)
            else:predictions=[];parent_graph=dict(parents=[],unknown_closure_not_counted_as_pass=True)
        budget();write(folder/'predictions.json',predictions);write(folder/'parent_graph.json',parent_graph)
        write(folder/'prediction_freeze.json',dict(source=args.source,status='PREDICTIONS_FROZEN',predicted=len(predictions),
            predictions_sha256=digest(folder/'predictions.json'),parent_graph_sha256=digest(folder/'parent_graph.json'),
            checkpoint_file=args.checkpoint,checkpoint_sha256=digest(OUT/args.training_run/args.checkpoint),neural_tokens=tokens,query_positions=provider.query_count if provider else 0,
            elapsed_seconds=time.perf_counter()-start,invalid_intervals=invalid_intervals,target_reads=0,
            cached_pipeline_not_contract_timing=True,full_contract_pass=False,end_utc=utc()))
        print('HIERARCHICAL_PREDICTION',args.source,'OUTPUTS',len(predictions),'SECONDS',round(time.perf_counter()-start,3),flush=True)
    except BaseException:
        write(folder/'failure.json',dict(error=traceback.format_exc(),tokens=tokens,
            query_positions=provider.query_count if provider else 0,elapsed_seconds=time.perf_counter()-start,utc=utc()));raise


if __name__=='__main__':main()
