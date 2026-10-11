"""Stream all original finite carriers into an input-only parent-object registry."""
import gc,json,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier

def stream_array(path):
    decoder=json.JSONDecoder();buffer='';position=0;started=False;finished=False
    with path.open(encoding='utf-8-sig') as f:
        while True:
            chunk=f.read(65536)
            if not chunk:finished=True
            buffer=buffer[position:]+chunk;position=0
            while True:
                while position<len(buffer) and buffer[position].isspace():position+=1
                if not started:
                    if position==len(buffer):break
                    if buffer[position]!='[':raise ValueError('Expected top-level profile array')
                    started=True;position+=1;continue
                if position>=len(buffer):break
                if buffer[position]==',':position+=1;continue
                if buffer[position]==']':return
                try:value,end=decoder.raw_decode(buffer,position)
                except json.JSONDecodeError:
                    if finished:raise
                    break
                yield value;position=end
            if finished:raise ValueError('Unterminated profile array')

def main():
    dest=OUT/'run251';dest.mkdir(exist_ok=False);rows=[];start=time.perf_counter()
    for key in ('21','BM79A_native_local','10','64'):
        check_resources();base=OUT/f'run185/source_{key}';freeze=read(base/'input_freeze.json')
        profiles_path=base/'profiles.json';assert digest(profiles_path)==freeze['files']['profiles.json']
        db=next(b for b in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases'] if b['source_key']==key)
        assert digest(OUT/'run201'/db['mapping_file'])==db['mapping_sha256']
        with np.load(OUT/'run201'/db['mapping_file']) as z:family=z['profile_family'];relation=z['profile_relation'];transform=z['local_to_canonical']
        n=len(family)
        if n>200000:raise RuntimeError('All-object input exceeds existing 200000 neural-token budget; no truncation')
        length=np.empty(n,np.float64);kind=np.empty(n,np.int8);closed=np.zeros(n,bool);face_pairs=[];side_counts={}
        folder=dest/f'source_{key}';folder.mkdir()
        with (folder/'object_provenance.jsonl').open('w',encoding='utf-8') as out:
            count=0
            for i,p in enumerate(stream_array(profiles_path)):
                assert i<n;curve=CurveCarrier(p['representation']);length[i]=curve.length;kind[i]=1 if curve.kind=='line' else 2
                closed[i]=bool(curve.kind=='arc' and curve.arc.is_closed);assert np.isfinite(length[i]) and length[i]>0
                side=p.get('weld_side',p.get('side_hypothesis'));side_key=json.dumps(side,sort_keys=True);side_counts[side_key]=side_counts.get(side_key,0)+1
                out.write(json.dumps(dict(object_id=i,original_profile=i,family=int(family[i]),relation=int(relation[i]),
                    ordered_face_indices=p.get('face_indices',[]),side_hypothesis=side,material_identity='UNKNOWN',
                    local_domain_mm=[0.,float(length[i])],kind=curve.kind,closed=bool(closed[i]),
                    support_intervals_mm=p.get('support_intervals_mm',[]),unobserved_gaps_mm=p.get('unobserved_gaps_mm',[])),ensure_ascii=False)+'\n')
                count+=1
            assert count==n
        np.savez(folder/'objects.npz',profile=np.arange(n,dtype=np.int64),family=family,relation=relation,length_mm=length,kind=kind,closed=closed,local_to_family=transform)
        cache=OUT/'run224/input_domain.npz' if key=='21' else OUT/f'run223/source_{key}/token_domain.npz'
        if cache.exists():
            with np.load(cache) as z:cached_families=np.unique(z['family'])
            covered=int(np.isin(family,cached_families).sum())
        else:cached_families=[];covered=0
        row=dict(source=key,all_original_profiles=n,objects=n,line_objects=int((kind==1).sum()),arc_objects=int((kind==2).sum()),
            closed_objects=int(closed.sum()),side_hypothesis_counts=side_counts,input_sha256=freeze['input_sha256'],
            object_file=f'source_{key}/objects.npz',object_sha256=digest(folder/'objects.npz'),
            provenance_sha256=digest(folder/'object_provenance.jsonl'),profiles_sha256=digest(profiles_path),
            old_cell_cache_file=str(cache.relative_to(OUT)),old_cached_families=len(cached_families),profiles_with_family_in_old_cell_cache=covered,
            old_cache_full_object_coverage=covered==n,bins_ready=False,anchors_ready=False,target_reads=0)
        rows.append(row);write(dest/'progress.json',dict(cases=rows,utc=utc()));print('ALL_PARENT_OBJECTS',key,n,'OLD_CACHE_COVERAGE',covered,'/',n,flush=True)
        del family,relation,transform,length,kind,closed;gc.collect()
    write(dest/'receipt.json',dict(status='ALL_ORIGINAL_CARRIERS_REGISTERED_FEATURES_PENDING',cases=rows,
        target_reads=0,STEP_reads=0,optimizer_updates=0,all_relations_and_sides_retained=True,object_count_is_not_predicted_weld_count=True,
        full_contract_pass=False,seconds=time.perf_counter()-start,utc=utc()))

if __name__=='__main__':main()
