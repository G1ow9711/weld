"""One unified score before interval DP, with frozen downstream model fields."""
import argparse,gc,json,pickle,time
import hashlib
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_unified_interval_workset_1288 import load_runtime,evidence,scalars,choice
DEST=OUT/'run458'

def dump(path,data):
    with path.open('w',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False,separators=(',',':'),
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())

def proposals(key):
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import tensors,select_intervals,CHUNK
    from weld_pipeline.sjg_finite_face_domains_1235 import ContinuousGraphBank
    from weld_pipeline.sjg_selected_interval_decoder_1296 import decode_selected_shared
    from weld_pipeline.sjg_same_interval_diagnostic_1285 import feature_matrix
    begin=time.perf_counter();check_resources();rt=load_runtime(key)
    tr=read(DEST/'training_receipt.json');assert digest(DEST/'unified_model.npz')==tr['model_sha256']
    with np.load(DEST/'unified_model.npz') as z:theta=z['theta']
    old_byrow={}
    for old in read(OUT/f'run432/proposal_{key}/selected_intervals.json')['intervals']:old_byrow.setdefault(old['row'],[]).append(old)
    choices=[];intervals=[];xx=[];metadata=[];total=0;parity_count=0;project_error=0.
    for number,r in enumerate(rt['records']):
        n=r['n'];a=r['q_offset'];row=r['row'];rawq=rt['s']['m']['q_mm'][a:a+n]
        if n<2:continue
        x,c,q,p=tensors(rt['s'],r,rt['payload']['norm'],rt['device']);model=rt['model']
        ev=evidence(rt,r);proj=ev.project(theta[:,360:600].reshape(4,4,60))
        endpoints=x.cpu().numpy();left=endpoints[:n]@theta[:,:180].T;right=endpoints[n:]@theta[:,180:360].T
        margins=np.full((n,n),-np.inf,np.float64);categories=np.zeros((n,n),np.uint8)
        zerom=np.full((n,n),-np.inf,np.float32);zeroc=np.zeros((n,n),np.uint8)
        # Only chosen original margins are later recomputed sparsely; no new S replaces old metadata.
        with torch.no_grad():
            h=model.encode(x,c)
            for lo in range(0,n-1,CHUNK):
                hi=min(lo+CHUNK,n-1);score=model.scores(h[:n],h[n:],q,p,r['curve_length_mm'],r['is_closed'],lo,hi)
                original=(score[:,:,1:]-score[:,:,:1]).cpu().numpy()
                zerom[lo:hi]=original.max(-1);zeroc[lo:hi]=original.argmax(-1)+1
                ii,jj=np.where(np.arange(lo,hi)[:,None]<np.arange(n)[None,:]);ii+=lo
                for start in range(0,len(ii),8192):
                    i,j=ii[start:start+8192],jj[start:start+8192]
                    inside=proj.scores(rawq[i],rawq[j])
                    if number<3 and start==0:
                        error=float(abs(inside[:3]-ev.features(rawq[i[:3]],rawq[j[:3]])@theta[:,360:600].T).max())
                        project_error=max(project_error,error);assert error<1e-7
                    residual=left[i]+right[j]+inside+scalars(rt,r,i,j)@theta[:,600:604].T+theta[:,604]
                    unified=original[i-lo,j].astype(np.float64)+residual
                    margins[i,j]=unified.max(-1);categories[i,j]=unified.argmax(-1)+1
        zero=select_intervals(zerom,zeroc);prior=old_byrow.get(row,[])
        assert len(zero)==len(prior),(key,row,len(zero),len(prior))
        for (i,j,cat,m),v in zip(zero,prior):
            assert (float(rawq[i]),float(rawq[j]),cat)==(v['lo_mm'],v['hi_mm'],v['category'])
            assert abs(m-v['margin'])<1e-4;parity_count+=1
        chosen=select_intervals(margins,categories)
        for i,j,cat,unified in chosen:
            with torch.no_grad():
                score=model.scores(h[:n],h[n:],q,p,r['curve_length_mm'],r['is_closed'],i,i+1)
                oldmargin=float((score[0,j,cat]-score[0,j,0]).cpu())
            picked=choice(rt,r,i,j,cat,len(choices),oldmargin)
            picked.update(unified_score=float(unified),original_query_key=f'{key}:{row}:{i}:{j}:{cat}')
            choices.append(picked)
            intervals.append(dict(row=row,lo_mm=float(rawq[i]),hi_mm=float(rawq[j]),category=cat,
                margin=oldmargin,unified_score=unified))
            xx.append(feature_matrix(rt['s'],r,np.array([i]),np.array([j]),np.array([cat]),np.array([oldmargin]))[0])
            faces=rt['graph'].profiles[row//2].get('face_indices',[])[:2];faces=faces[::-1] if row%2 else faces
            metadata.append(dict(graph_nodes=[],face_pairs=[faces],physical_sides=['UNKNOWN'],side_known=False,
                probability_mean=float(np.exp(-np.logaddexp(0.,-oldmargin))),length_mm=float(rawq[j]-rawq[i]),
                line_fraction=float(rt['graph'].curve(row//2).kind=='line')))
        total+=n*(n-1)//2
        if number%100==0:
            print('FULL_DOMAIN',key,'CARRIERS',number+1,'/',len(rt['records']),'SELECTED',len(choices),flush=True)
            check_resources()
        del x,c,q,p,h,margins,categories,zerom,zeroc,ev,proj
    manifest=dict(schema='FROZEN_UNIFIED_DP_SELECTION_V1',model_sha256=tr['model_sha256'],
        input_binding=rt['s']['receipt'],DP_implementation_sha256=digest(Path(__file__)),
        selection_sha256=hashlib.sha256(json.dumps(intervals,sort_keys=True).encode()).hexdigest(),choices=choices)
    result=decode_selected_shared(choices,ContinuousGraphBank(rt['graph']),manifest)
    result.update(GT_reads=0,STEP_reads=0)
    folder=DEST/f'trained/proposal_{key}';folder.mkdir(parents=True)
    dump(folder/'decoded.json',result);write(folder/'selected_intervals.json',dict(intervals=intervals))
    write(folder/'selected_manifest.json',manifest)
    assert result['total_prediction_children']==len(intervals)
    np.save(folder/'features526.npy',np.asarray(xx,np.float32).reshape(-1,526));write(folder/'metadata.json',metadata)
    write(folder/'prediction_receipt.json',dict(source=key,decoded_sha256=digest(folder/'decoded.json'),
        features_sha256=digest(folder/'features526.npy'),metadata_sha256=digest(folder/'metadata.json'),
        all_pairs_scored=total,old_DP_intervals_exact=parity_count,projection_error=project_error,
        proposed_intervals=len(choices),GT_reads=0,STEP_reads=0,zero_residual_DP_matches_frozen432=True,
        old_margin_probability_fields_preserved=True,full_contract_pass=False,seconds=time.perf_counter()-begin))
    print('UNIFIED_PROPOSALS_COMPLETE',key,'PAIRS',total,'SELECTED',len(choices),flush=True)

def postprocess(key,arm):
    from weld_pipeline.sjg_additive_parent_precision_1254 import source,endpoint_distances,candidate_pair_possible,pair_features
    from weld_pipeline.sjg_child_parent_supervision_1255 import singleton
    from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
    from weld_pipeline.sjg_child_parent_model_1256 import assemble
    begin=time.perf_counter();check_resources();route=read(OUT/'run433/inputs_frozen_before_train_labels.json')['routes'][key]
    route=dict(route,proposal_run='run432' if arm=='zero' else 'run458/trained')
    baseline,additions,bm,binding=source(key,route)
    if arm=='zero':
        x=np.load(OUT/f'run434/source_{key}/features.npy')
        metadata=read(OUT/f'run433/source_{key}/metadata.json')[len(baseline['parents']):]
    else:
        folder=DEST/f'trained/proposal_{key}';receipt=read(folder/'prediction_receipt.json')
        assert digest(folder/'features526.npy')==receipt['features_sha256'] and digest(folder/'metadata.json')==receipt['metadata_sha256']
        x=np.load(folder/'features526.npy');metadata=read(folder/'metadata.json')
    intervals=read(OUT/route['proposal_run']/f'proposal_{key}/selected_intervals.json')['intervals']
    query_ids=np.asarray([p['predicted_parent_query'] for p in additions['parents']],dtype=int)
    invalid_ids=[v['parent_query'] for v in additions['invalid_predictions']]
    assert sorted(query_ids.tolist()+invalid_ids)==list(range(len(intervals)))
    assert len(x)==len(metadata)==len(intervals)
    x=x[query_ids];metadata=[metadata[i] for i in query_ids];intervals=[intervals[i] for i in query_ids]
    scores=np.asarray([v['margin'] if arm=='zero' else v['unified_score'] for v in intervals]);assert np.all(scores>0)
    parents=[];allmeta=[]
    for parent,m in zip(baseline['parents'],bm):
        for seg in parent['segments']:
            parents.append(singleton(seg));allmeta.append(dict(m,length_mm=seg['length_mm'],line_fraction=float(seg['kind']=='line')))
    base=len(parents);parents.extend(additions['parents']);allmeta.extend(metadata)
    context=ParentRelations(parents,allmeta)
    with (OUT/'run446/models.pkl').open('rb') as f:models=pickle.load(f)['models']
    probability=models['validity'].predict_proba(x)[:,1] if len(x) else np.empty(0) # recorded only; never vetoes
    distance=endpoint_distances(parents);selected=list(range(base));accepted=[];decisions=[]
    for local in sorted(range(len(scores)),key=lambda i:(-float(scores[i]),-float(x[i,523]),i)):
        index=base+local;duplicate=None;close=[j for j in selected if candidate_pair_possible(context,distance,index,j)]
        if close:
            q=models['duplicate'].predict_proba(np.stack([pair_features(context,index,j) for j in close]))[:,1];j=int(q.argmax())
            if q[j]>.5:duplicate=dict(selected_child_index=close[j],probability=float(q[j]))
        if duplicate is None:selected.append(index);accepted.append(local)
        decisions.append(dict(candidate_index=local,original_query_id=int(query_ids[local]),unified_score=float(scores[local]),
            original_validity_probability=float(probability[local]),validity_probability_used_as_gate=False,
            duplicate_of=duplicate,selected=duplicate is None))
    folder=DEST/f'{arm}/test_{key}';folder.mkdir(parents=True)
    remaining=ParentRelations([parents[base+i] for i in accepted],[allmeta[base+i] for i in accepted])
    assembled=assemble(remaining,models['link'],folder)
    result=dict(baseline,parents=baseline['parents']+assembled,
        invalid_predictions=baseline['invalid_predictions']+additions['invalid_predictions'])
    result['total_prediction_children']=sum(p['child_count'] for p in result['parents'])+sum(p['counted_false_children'] for p in result['invalid_predictions'])
    result.update(GT_reads=0,STEP_reads=0);dump(folder/'final_decoded.json',result)
    write(folder/'decisions.json',dict(decisions=decisions,accepted_child_indices=accepted,valid_query_ids=query_ids.tolist(),invalid_query_ids=invalid_ids))
    write(folder/'prediction_receipt.json',dict(source=key,decoded_sha256=digest(folder/'final_decoded.json'),**binding,
        raw_proposal_path=str(OUT/route['proposal_run']/f'proposal_{key}/decoded.json'),GT_reads=0,STEP_reads=0,old_forests_frozen=True,old_field_meanings_preserved=True,independent_D_gate=False,
        duplicate_priority='unified_score_then_frozen_old_margin_then_index',full_contract_pass=False,seconds=time.perf_counter()-begin))
    print('PARENT_OUTPUT',arm,key,'RAW',len(scores),'RETAINED',len(accepted),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['proposals','post'],required=True)
    p.add_argument('--source',required=True);p.add_argument('--arm',choices=['zero','trained'],default='trained');a=p.parse_args()
    proposals(a.source) if a.stage=='proposals' else postprocess(a.source,a.arm)
