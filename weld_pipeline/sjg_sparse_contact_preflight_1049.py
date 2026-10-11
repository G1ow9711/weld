"""Freeze all nine pure-input token domains, then audit TRAIN event capacity."""
import argparse,gc,time,traceback,shutil
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_balanced_hierarchy_1043 import load_model
from weld_pipeline.sjg_sparse_contact_domain_1048 import build,controls
from weld_pipeline.sjg_hierarchical_data_1030 import family_supervision
from weld_pipeline.sjg_projected_boundary_anchors_1050 import ProjectedBoundaryAnchors,controls as boundary_controls

def capacity(domain,arrays,supervised):
    covered=set();overflow=[];outside=[];slot_counts=[]
    for fid in np.unique(arrays['family']):
        fid=int(fid);ids=np.flatnonzero(arrays['family']==fid);cells=arrays['cells'][ids,:2];events=[set() for _ in ids]
        children=supervised.get(fid,{})
        for child,meta in children.items():
            for left,right in meta['pieces']:
                overlap=np.maximum(0,np.minimum(cells[:,1],right)-np.maximum(cells[:,0],left)).sum()
                if overlap<right-left-1e-7:outside.append(dict(family=fid,child=child,reason='reference interval outside token domain'))
            if meta['closed']:continue
            points=[v for piece in meta['pieces'] for v in piece]
            circ=domain['families'][fid]['circumference_mm']
            if circ and len(meta['pieces'])==2:
                points=[v for v in points if abs(v)>1e-8 and abs(v-circ)>1e-8]
            for point in points:
                j=int(np.searchsorted(cells[:,0],point,side='right')-1)
                if j<0 or point>cells[j,1]+1e-8:outside.append(dict(family=fid,child=child,point=float(point),reason='event outside cell'));continue
                # Numerical equality only, never the 3 mm acceptance tolerance.
                events[j].add(round(float(point),8))
        bad=[int(ids[j]) for j,e in enumerate(events) if len(e)>2]
        if bad:overflow.append(dict(family=fid,cells=bad,counts=[len(events[int(i-ids[0])]) for i in bad]))
        if not bad and not any(x['family']==fid for x in outside):covered.update(children)
        slot_counts.extend(len(e) for e in events)
    return dict(covered_ids=sorted(covered),overflow_families=overflow,outside_domains=outside,
        maximum_distinct_events_in_one_cell=max(slot_counts,default=0),event_equality_tolerance_mm=1e-8)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--projected-boundaries',action='store_true');a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    (dest/'source_snapshot').mkdir()
    for n in (1048,1049,1050):
        for p in (ROOT/'weld_pipeline').glob('*_'+str(n)+'.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    start=time.perf_counter();write(dest/'controls.json',dict(domain=controls(),boundary=boundary_controls()))
    write(dest/'registered_protocol.json',dict(stage='REPLACEMENT_SPARSE_CONTACT_DOMAIN_PREFLIGHT',frozen_selector='run218/final.pt',
        all_nine_inputs_before_TRAIN_labels=True,heldout_supervision=False,all_G=500,semantic_token_limit=100000,XYZ_query_limit=100000,
        boundary_slots_per_cell=2,projected_boundary_anchors=a.projected_boundaries,optimizer_updates=0,no_anchors_from_GT=True,full_contract_pass=False,start_utc=utc()))
    try:
        torch.set_num_threads(2);checkpoint=OUT/'run218/final.pt';model=load_model(torch.load(checkpoint,map_location='cpu'));model.eval()
        cases=read(OUT/'run98/local_cohort.json')['cases'];bindings={r['source_key']:r for r in read(OUT/'run206/training_manifest.json')['static_bindings']}
        domains={r['source_key']:r for r in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases']};frozen=[]
        for case in cases:
            check_resources();key=case['source_key'];folder=dest/('source_'+key);folder.mkdir();b=bindings[key];db=domains[key]
            assert digest(OUT/'run206'/b['file'])==b['sha256'] and digest(OUT/'run201'/db['json_file'])==db['json_sha256']
            assert digest(OUT/'run201'/db['mapping_file'])==db['mapping_sha256']
            with np.load(OUT/'run206'/b['file']) as z:s={k:torch.from_numpy(z[k]) for k in z.files}
            with torch.inference_mode():
                r,rl=model.relations(s['relation_x']);f,fl=model.families(r,s['family_relation'],s['family_x']);rp=rl.sigmoid().numpy();fp=fl.sigmoid().numpy()
            passed=rp>=.5;tested=passed[s['family_relation'].numpy()];selected=np.flatnonzero(tested&(fp>=.5))
            np.savez(folder/'selector_features.npz',family_ids=selected,family_latent=f[selected].numpy(),relation_score=rp,family_score=fp)
            domain=read(OUT/'run201'/db['json_file']);base=OUT/f'run185/source_{key}';input_binding=read(base/'input_freeze.json')
            assert digest(base/'profiles.json')==input_binding['files']['profiles.json'] and input_binding['input_sha256']==case['input_sha256']
            profiles=read(base/'profiles.json')
            with np.load(OUT/'run201'/db['mapping_file']) as z:transforms=z['local_to_canonical']
            if a.projected_boundaries:
                fb=read(OUT/f'run166/source_{key}_freeze.json');assert fb['input_sha256']==case['input_sha256']
                feature_file=OUT/'run166'/fb['feature_file'];assert digest(feature_file)==fb['feature_sha256']
                assert digest(OUT/case['input_file'])==case['input_sha256']
                xyz=np.load(OUT/case['input_file'],mmap_mode='r',allow_pickle=False)
                with np.load(feature_file) as z:boundary=z['boundary']
                assert digest(base/'surfaces.json')==input_binding['files']['surfaces.json']
                assert digest(base/'surface_support.npz')==input_binding['files']['surface_support.npz']
                faces=read(base/'surfaces.json')
                with np.load(base/'surface_support.npz') as support:
                    provider=ProjectedBoundaryAnchors(xyz,boundary,faces,support)
                    meta,arrays=build(domain,profiles,transforms,selected,provider)
                del xyz,boundary,faces,provider
            else:meta,arrays=build(domain,profiles,transforms,selected)
            meta.update(source=key,relation_stage_tokens=len(rp),family_stage_tokens=int(tested.sum()),
                total_neural_tokens=len(rp)+int(tested.sum())+meta['semantic_tokens'])
            meta['budget_pass']=meta['total_neural_tokens']<=100000 and meta['unique_XYZ_queries']<=100000
            np.savez(folder/'token_domain.npz',**arrays);write(folder/'token_domain.json',meta)
            row=dict(source=key,files={n:digest(folder/n) for n in ('selector_features.npz','token_domain.npz','token_domain.json')},
                semantic_tokens=meta['semantic_tokens'],total_neural_tokens=meta['total_neural_tokens'],queries=meta['unique_XYZ_queries'],budget_pass=meta['budget_pass'])
            frozen.append(row);write(dest/'progress.json',dict(frozen_inputs=frozen))
            print('SPARSE_DOMAIN_FROZEN',key,'TOKENS',row['semantic_tokens'],'TOTAL',row['total_neural_tokens'],'QUERIES',row['queries'],'BUDGET',row['budget_pass'],flush=True)
            del s,r,rl,f,fl,rp,fp,domain,profiles,arrays,transforms;gc.collect()
        write(dest/'all_inputs_frozen_before_labels.json',dict(cases=frozen,checkpoint_sha256=digest(checkpoint),target_reads=0))
        train_keys={r['source_key'] for r in read(OUT/'run206/training_manifest.json')['cases']};reports=[]
        for case in cases:
            key=case['source_key']
            if key not in train_keys:continue
            folder=dest/('source_'+key);db=domains[key];domain=read(OUT/'run201'/db['json_file'])
            assert digest(OUT/case['target_file'])==case['target_sha256'];target=read(OUT/case['target_file'])
            with np.load(OUT/'run201'/db['mapping_file']) as z:pf=z['profile_family'];transforms=z['local_to_canonical']
            supervised,_=family_supervision(key,domain,pf,transforms,target)
            with np.load(folder/'token_domain.npz') as z:arrays={k:z[k] for k in z.files}
            result=capacity(domain,arrays,supervised);children=[c for p in target['parents'] for c in p['children']];all_ids={c['original_child_id'] for c in children}
            arcs={c['original_child_id'] for c in children if c['representation']['kind']=='arc'};covered=set(result['covered_ids'])&all_ids
            result.update(source=key,G=len(all_ids),covered_G=len(covered),missing=sorted(all_ids-covered),arc_G=len(arcs),covered_arcs=len(arcs&covered))
            write(folder/'TRAIN_capacity.json',result);reports.append(result)
            print('SPARSE_CAPACITY',key,len(covered),'/',len(all_ids),'OVERFLOW_FAMILIES',len(result['overflow_families']),flush=True)
            del domain,target,arrays,supervised,pf,transforms;gc.collect()
        passed=all(r['budget_pass'] for r in frozen) and sum(r['covered_G'] for r in reports)==363 and all(not r['overflow_families'] and not r['outside_domains'] for r in reports)
        write(dest/'receipt.json',dict(status='SPARSE_CONTACT_PREFLIGHT_PASSED' if passed else 'SPARSE_CONTACT_PREFLIGHT_FAILED',
            inputs=frozen,TRAIN_G=363,covered_G=sum(r['covered_G'] for r in reports),covered_arcs=sum(r['covered_arcs'] for r in reports),
            overflow_family_count=sum(len(r['overflow_families']) for r in reports),outside_event_count=sum(len(r['outside_domains']) for r in reports),
            optimizer_updates=0,all_G=500,full_contract_pass=False,seconds=time.perf_counter()-start,end_utc=utc()))
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),utc=utc()));raise

if __name__=='__main__':main()
