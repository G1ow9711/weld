"""Pure predicted finite-face/material partitions across original XYZ blocks."""
import argparse
import gc
import inspect
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_overlap_instances_973 import local_partition,stitch,controls


def main(partitioner=local_partition,method='farthest_cover',subset=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);source=OUT/'run165';rows=[]
    assert read(source/'receipt.json')['status']=='NATIVE_MATERIAL_SURFACE_FEATURES_FROZEN'
    manifest=read(source/'all_predictions_frozen.json');assert len(manifest['cases'])==9
    cases=manifest['cases'] if subset is None else [c for c in manifest['cases'] if c['source_key'] in subset]
    if subset is not None:assert {c['source_key'] for c in cases}==set(subset)
    write(dest/'registered_protocol.json',dict(stage='NATIVE_OVERLAP_INSTANCE_HYPOTHESES',feature_run='run165',
        feature_manifest_sha256=digest(source/'all_predictions_frozen.json'),source_sha256=digest(__file__),
        partition_source_sha256=digest(__file__.replace('sjg_native_instance_freeze_974','sjg_overlap_instances_973')),
        partition_method=method,partition_implementation_sha256=digest(inspect.getfile(partitioner)),
        selected_sources=[c['source_key'] for c in cases],subset_diagnostic=subset is not None,
        density_modes_parameters=dict(seeds=128,iterations=12,min_mode_points=8) if method=='density_modes' else None,
        local_embedding_radius=.4,merge_min_votes=16,merge_min_agreement=.9,
        cross_block_embedding_distances=False,same_block_distinct_instance_merge_veto=True,
        all_original_points_preserved=True,original_eight_G=496,supplemental_BM_G=4,
        grouping_is_prediction_not_material_truth=True,multi_owner_contact_assignment_pending=True,
        label_and_STEP_reads=0,full_contract_pass=False,start_utc=utc()))
    write(dest/'controls.json',controls())
    try:
        for case in cases:
            check_resources();key=case['source_key'];folder=dest/('source_'+key);folder.mkdir()
            n=case['original_points'];seen=np.zeros(n,np.uint8)
            core_nodes=[np.full(n,-1,np.int32),np.full(n,-1,np.int32)]
            node_blocks=[[],[]];local_rows=[];started=time.perf_counter()
            for bi,entry in enumerate(case['blocks']):
                if bi%20==0:check_resources()
                path=source/entry['file'];assert digest(path)==entry['sha256']
                with np.load(path) as z:
                    core=z['original_point_ids'];context=z['context_original_point_ids'];ids=np.r_[core,context]
                    assert len(core)==entry['core_points'] and len(np.unique(ids))==len(ids)
                    assert np.all((ids>=0)&(ids<n));np.add.at(seen,core,1)
                    record=dict(original_point_ids=ids,core_count=np.array(len(core),np.int64))
                    for h,field in enumerate(('embedding','material_embedding')):
                        value=np.r_[z[field],z['context_'+field]];labels=partitioner(value)
                        count=int(labels.max())+1;nodes=labels+len(node_blocks[h]);node_blocks[h].extend([bi]*count)
                        core_nodes[h][core]=nodes[:len(core)];record[('face_node','material_node')[h]]=nodes
                file=folder/f'nodes_{bi:04d}.npz';np.savez(file,**record)
                local_rows.append(dict(file=str(file.relative_to(dest)),sha256=digest(file),input_feature_sha256=entry['sha256']))
                if bi%20==0 or bi+1==len(case['blocks']):
                    write(dest/'progress.json',dict(completed_cases=len(rows),source=key,block=bi+1,blocks=len(case['blocks']),time_utc=utc()))
                    print('LOCAL_INSTANCE_BLOCK',key,bi+1,'/',len(case['blocks']),'NODES',*[len(b) for b in node_blocks],flush=True)
            assert np.all(seen==1) and all(np.all(v>=0) for v in core_nodes)
            output=dict(core_coverage=seen);stats={}
            for h,field in enumerate(('face_node','material_node')):
                count=len(node_blocks[h]);codes=[]
                for entry in local_rows:
                    with np.load(dest/entry['file']) as z:
                        core=int(z['core_count']);ids=z['original_point_ids'][core:]
                        a=z[field][core:].astype(np.int64);b=core_nodes[h][ids].astype(np.int64)
                    codes.append(np.minimum(a,b)*count+np.maximum(a,b))
                code,votes=np.unique(np.concatenate(codes),return_counts=True);del codes
                pairs=np.column_stack((code//count,code%count))
                labels,detail=stitch(node_blocks[h],pairs,votes)
                name=('face_instance','material_instance')[h];output[name]=labels[core_nodes[h]]
                sizes=np.bincount(output[name]);detail.update(local_nodes=count,global_instances=len(sizes),
                    largest_instance_points=int(sizes.max()),singleton_points=int((sizes==1).sum()),
                    points_in_instances_under_16=int(sizes[sizes<16].sum()))
                stats[name]=detail
                np.savez(folder/(name+'_overlap.npz'),pairs=pairs,votes=votes,node_block=np.asarray(node_blocks[h],np.int32),node_global=labels)
                del code,votes,pairs,labels,sizes
            file=dest/f'source_{key}_instances.npz';np.savez(file,**output)
            row=dict(source_key=key,role=case['role'],original_points=n,input_sha256=case['input_sha256'],
                file=file.name,sha256=digest(file),blocks=local_rows,statistics=stats,
                all_original_points_covered_once=True,elapsed_seconds=time.perf_counter()-started)
            rows.append(row);write(folder/'freeze.json',row)
            print('NATIVE_INSTANCES_FROZEN',key,stats,flush=True)
            del output,core_nodes,node_blocks,seen,record,value;gc.collect()
        write(dest/'all_predictions_frozen.json',dict(sources=rows,label_and_STEP_reads=0))
        write(dest/'receipt.json',dict(status='NATIVE_NINE_INSTANCE_HYPOTHESES_FROZEN' if subset is None else 'NATIVE_SUBSET_INSTANCE_HYPOTHESES_FROZEN',sources=rows,
            full_contract_pass=False,weld_accuracy_not_measured=True,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_cases=len(rows),error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
