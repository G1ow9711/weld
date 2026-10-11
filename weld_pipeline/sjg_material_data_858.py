"""TRAIN-only material-reference targets on the existing input-selected PTv3 blocks."""
import argparse
import numpy as np
from weld_pipeline.sjg_prepare_776 import OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources


def mask_voxel_conflicts(owner,inverse,voxel_count):
    owner=np.asarray(owner,np.int64).copy();known=owner>=0
    low=np.full(voxel_count,np.iinfo(np.int64).max,np.int64);high=np.full(voxel_count,-1,np.int64)
    np.minimum.at(low,inverse[known],owner[known]);np.maximum.at(high,inverse[known],owner[known])
    conflict=low<high;owner[conflict[inverse]]=-1
    return owner,conflict[inverse]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);base=OUT/'run26';manifest=read(base/'block_manifest.json')
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};records=[];rows=[]
    write(dest/'registered_protocol.json',dict(stage='TRAIN_MATERIAL_BLOCK_TARGETS',train=TRAIN,
        development_not_read=VALID,arc_not_read=ARC,input_blocks='run26',input_block_manifest_sha256=digest(base/'block_manifest.json'),
        no_new_input_partition=True,no_reference_owner_as_input=True,ambiguous_points_unknown=True,
        conflicting_voxel_owners_unknown=True,reference_owners_not_customer_approved_truth=True,
        source_sha256=digest(__file__),full_contract_pass=False,start_utc=utc()))
    for source in TRAIN:
        check_resources();case=manifest['cases'][str(source)];teacher=OUT/'run22'/teachers[source]['teacher_file']
        assert digest(teacher)==teachers[source]['teacher_sha256'];assert case['teacher_sha256']==digest(teacher)
        mapping=base/f'blocks_{source:02d}/input_blocks.npz';assert digest(mapping)==case['mapping_sha256']
        with np.load(teacher) as z:owner=z['reference_volume'];ambiguous=z['multiple_material']
        assert len(owner)==case['points'] and np.all(owner[ambiguous]<0)
        seen=np.zeros(len(owner),np.uint8);known_core=conflict_core=0;folder=dest/f'source_{source:02d}';folder.mkdir()
        with np.load(mapping) as trace:
            for bi,binding in enumerate(case['blocks']):
                path=base/binding['path'];assert digest(path)==binding['sha256']
                ids=trace[f'ids_{bi}'].astype(np.int64);count=int(trace[f'core_count_{bi}'])
                assert len(np.unique(ids))==len(ids);seen[ids[:count]]+=1
                with np.load(path) as z:
                    inverse=z['x_inverse'].astype(np.int64);voxel_count=len(z['x_coord'])
                    assert np.array_equal(z['y_core'],np.arange(len(ids))<count)
                assert len(inverse)==len(ids)
                labels,conflict=mask_voxel_conflicts(owner[ids],inverse,voxel_count)
                file=folder/f'block_{bi:04d}_material.npy';np.save(file,labels.astype(np.int32))
                known_core+=int((labels[:count]>=0).sum());conflict_core+=int(conflict[:count].sum())
                records.append(dict(source=source,block=bi,input_file=str(path.relative_to(OUT)),input_sha256=binding['sha256'],
                    material_file=str(file.relative_to(dest)),material_sha256=digest(file),core_points=count,
                    known_owner_core_points=int((labels[:count]>=0).sum()),voxel_conflict_core_points=int(conflict[:count].sum())))
        assert np.all(seen==1)
        row=dict(source=source,original_points=len(owner),blocks=len(case['blocks']),known_owner_core_points=known_core,
            unknown_owner_core_points=len(owner)-known_core,voxel_conflict_core_points=conflict_core,
            raw_ambiguous_material_points=int(ambiguous.sum()),reference_owners=len(np.unique(owner[owner>=0])),
            every_original_point_has_one_core=True,teacher_sha256=digest(teacher),mapping_sha256=digest(mapping))
        rows.append(row);print('MATERIAL_TARGETS',row,flush=True)
    assert len(records)==len(manifest['train_schedule'])
    write(dest/'manifest.json',dict(records=records,rows=rows,only_TRAIN=True))
    write(dest/'receipt.json',dict(status='TRAIN_MATERIAL_BLOCK_TARGETS_COMPLETE',blocks=len(records),
        original_points=sum(r['original_points'] for r in rows),known_owner_core_points=sum(r['known_owner_core_points'] for r in rows),
        unknown_owner_core_points=sum(r['unknown_owner_core_points'] for r in rows),full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
