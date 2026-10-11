"""Prepare input-only blocks on CPU, without loading Torch or whole-case features."""
import argparse
import gc
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources

def core_partitions(xyz,maximum=4096):
    pending=[np.arange(len(xyz),dtype=np.int64)]
    while pending:
        ids=pending.pop()
        if len(ids)<=maximum:yield ids;continue
        axis=int(np.argmax(np.ptp(xyz[ids],axis=0)))
        order=np.argsort(xyz[ids,axis],kind='stable');half=len(ids)//2
        pending.extend([ids[order[:half]],ids[order[half:]]])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);cohort=read(OUT/'run98/local_cohort.json');records=[]
    assert cohort['coordinate_frame']=='workpiece_local' and cohort['unit']=='mm'
    write(dest/'registered_protocol.json',dict(stage='CPU_NATIVE_LOCAL_BLOCK_PREPARATION',start_utc=utc(),
        source_sha256=digest(__file__),cohort_sha256=digest(OUT/'run98/local_cohort.json'),
        numerical_partition_rule='same longest-axis stable split as 807, core<=4096, context to 8192',
        all_input_points_preserved=True,label_or_STEP_reads=0,optimizer_updates=0,
        original_G=496,resource_guard_unchanged=True,replaces_failed_in_memory_run='run100'))
    try:
        for case in cohort['cases']:
            check_resources();path=OUT/case['input_file'];assert digest(path)==case['input_sha256']
            xyz=np.load(path,mmap_mode='r',allow_pickle=False);assert xyz.shape[1]==3 and np.isfinite(xyz).all()
            key=case['source_key'];folder=dest/f'source_{key}';folder.mkdir();tree=cKDTree(xyz)
            seen=np.zeros(len(xyz),np.uint8);blocks=[]
            for bi,core in enumerate(core_partitions(xyz)):
                if bi%20==0:check_resources()
                near=np.asarray(tree.query(xyz[core].mean(0),k=min(8192,len(xyz)))[1],np.int64).reshape(-1)
                context=near[~np.isin(near,core)][:8192-len(core)]
                ids=np.r_[core,context].astype(np.int64);seen[core]+=1
                file=folder/f'block_{bi:04d}.npz'
                np.savez(file,xyz_mm=xyz[ids],original_point_ids=ids,core_count=np.int64(len(core)))
                blocks.append(dict(file=str(file.relative_to(dest)),sha256=digest(file),core_count=len(core)))
                if bi%20==0:print('LOCAL_CPU_BLOCKS',key,'BLOCK',bi+1,'COVERED',int(seen.sum()),'/',len(xyz),flush=True)
            assert np.all(seen==1)
            row=dict(source_key=key,role=case['role'],input_file=case['input_file'],input_sha256=case['input_sha256'],
                coordinate_frame='workpiece_local',unit='mm',original_points=len(xyz),blocks=blocks)
            write(folder/'manifest.json',row);records.append(row)
            print('LOCAL_CPU_CASE_FROZEN',key,len(xyz),'BLOCKS',len(blocks),flush=True)
            del tree,xyz,seen;gc.collect()
        write(dest/'block_manifest.json',dict(cases=records,label_reads=0,all_original_points_covered_once=True))
        write(dest/'receipt.json',dict(status='LOCAL_STREAM_BLOCKS_READY',cases=len(records),
            points=sum(r['original_points'] for r in records),blocks=sum(len(r['blocks']) for r in records),
            optimizer_updates=0,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
