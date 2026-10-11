"""Bind original-index TRAIN teachers to immutable native XYZ-only blocks."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def mask_conflicts(labels,inverse,count):
    result={k:v.copy() for k,v in labels.items()};stats={}
    for key in ('instance','reference_volume','boundary_target'):
        y=result[key];known=y>=0;low=np.full(count,np.iinfo(np.int32).max,np.int32);high=np.full(count,-1,np.int32)
        np.minimum.at(low,inverse[known],y[known]);np.maximum.at(high,inverse[known],y[known])
        conflict=(low<high)[inverse];stats[key]=int((known&conflict&result['core']).sum());y[conflict]=-1
    result['normal'][result['instance']<0]=0
    return result,stats


def controls():
    inverse=np.array([0,0,1,1]);labels=dict(instance=np.array([1,2,3,3]),reference_volume=np.array([1,2,4,-1]),
        boundary_target=np.array([0,1,1,-1]),normal=np.ones((4,3)),core=np.ones(4,bool))
    result,_=mask_conflicts(labels,inverse,2)
    assert np.array_equal(result['instance'],[-1,-1,3,3])
    assert np.array_equal(result['reference_volume'],[-1,-1,4,-1])
    assert np.array_equal(result['boundary_target'],[-1,-1,1,-1])
    assert np.all(result['normal'][:2]==0)
    return dict(voxel_owner_and_face_conflicts_unknown=True,unknown_labels_never_filled=True,boundary_conflicts_unknown=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    assert read(OUT/'run160/receipt.json')['status']=='NATIVE_TRAIN_PLANAR_TEACHERS_READY'
    teachers=read(OUT/'run160/manifest.json')['rows'];blocks={c['source_key']:c for c in read(OUT/'run101/block_manifest.json')['cases']}
    write(dest/'registered_protocol.json',dict(stage='NATIVE_STRUCTURAL_TRAIN_BLOCK_BINDING',teacher_run='run160',block_run='run101',
        train=[r['source_key'] for r in teachers],development_not_read=['64','24'],grid_mm=1.,
        all_original_core_points_once=True,all_known_core_points_in_dense_losses=True,
        voxel_conflicts_masked_and_counted=True,coordinate_frame='workpiece_local',unit='mm',
        STEP_labels_not_input_features=True,source_sha256=digest(__file__),full_contract_pass=False,start_utc=utc()))
    write(dest/'controls.json',controls());records=[];sources=[]
    for teacher in teachers:
        key=teacher['source_key'];case=blocks[key]
        assert case['role']=='TRAIN' and case['input_sha256']==teacher['input_sha256']
        assert digest(OUT/teacher['input_file'])==teacher['input_sha256'] and digest(OUT/teacher['teacher_file'])==teacher['teacher_sha256']
        xyz=np.load(OUT/teacher['input_file'],mmap_mode='r')
        with np.load(OUT/teacher['teacher_file']) as z:labels={k:z[k] for k in ('instance','normal','reference_volume','boundary_target')}
        assert all(len(v)==len(xyz) for v in labels.values());seen=np.zeros(len(xyz),np.uint8)
        totals=dict(core_points=0,known_face_core=0,known_material_core=0,boundary_positive_core=0,boundary_negative_core=0)
        conflicts={k:0 for k in ('instance','reference_volume','boundary_target')}
        folder=dest/('source_'+key);folder.mkdir()
        for bi,b in enumerate(case['blocks']):
            if bi%32==0:check_resources()
            path=OUT/'run101'/b['file'];assert digest(path)==b['sha256']
            with np.load(path) as z:points=z['xyz_mm'];ids=z['original_point_ids'];core=int(z['core_count'])
            assert core==b['core_count'] and len(np.unique(ids))==len(ids) and np.array_equal(points,xyz[ids])
            np.add.at(seen,ids[:core],1)
            grid=np.floor(points-points.min(0)).astype(np.int32);unique,inverse=np.unique(grid,axis=0,return_inverse=True)
            y={k:v[ids] for k,v in labels.items()};y['core']=np.arange(len(ids))<core
            y,c=mask_conflicts(y,inverse,len(unique));file=folder/f'labels_{bi:04d}.npz';np.savez_compressed(file,**y)
            row=dict(source_key=key,input_file=str(path.relative_to(OUT)),input_sha256=b['sha256'],
                label_file=str(file.relative_to(OUT)),label_sha256=digest(file),core_count=core,
                points=len(ids),voxels=len(unique),teacher_sha256=teacher['teacher_sha256'])
            records.append(row);totals['core_points']+=core
            totals['known_face_core']+=int(((y['instance']>=0)&y['core']).sum())
            totals['known_material_core']+=int(((y['reference_volume']>=0)&y['core']).sum())
            totals['boundary_positive_core']+=int(((y['boundary_target']==1)&y['core']).sum())
            totals['boundary_negative_core']+=int(((y['boundary_target']==0)&y['core']).sum())
            for k in conflicts:conflicts[k]+=c[k]
            if bi%64==0:print('NATIVE_LABEL_BLOCK',key,bi+1,'/',len(case['blocks']),flush=True)
        assert np.all(seen==1) and totals['core_points']==teacher['points']
        row=dict(source_key=key,blocks=len(case['blocks']),original_points=len(xyz),conflicts=conflicts,**totals);sources.append(row)
        print('NATIVE_LABEL_SOURCE_COMPLETE',row,flush=True);del xyz,labels,seen;gc.collect()
    write(dest/'manifest.json',dict(records=records,sources=sources,teacher_manifest_sha256=digest(OUT/'run160/manifest.json'),
        input_block_manifest_sha256=digest(OUT/'run101/block_manifest.json')))
    write(dest/'receipt.json',dict(status='NATIVE_STRUCTURAL_TRAIN_BLOCKS_READY',blocks=len(records),sources=sources,
        core_points=sum(s['core_points'] for s in sources),known_face_core=sum(s['known_face_core'] for s in sources),
        known_material_core=sum(s['known_material_core'] for s in sources),original_eight_G=496,
        supplemental_BM_weld_TRAIN_unchanged=True,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
