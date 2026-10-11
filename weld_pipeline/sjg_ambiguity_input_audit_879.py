"""TRAIN-only diagnosis of sampling multiplicity lost by XYZ voxel features."""
import argparse
import numpy as np
from weld_pipeline.sjg_prepare_776 import OUT,TRAIN
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_ambiguity_prepare_875 import confusion
from weld_pipeline.sjg_ambiguity_evaluate_877 import scores


def histogram(counts,labels):
    return {name:dict(unique_owner=int((mask&(labels==0)).sum()),multi_owner=int((mask&(labels==1)).sum()),
        unknown=int((mask&(labels<0)).sum())) for name,mask in [('1',counts==1),('2',counts==2),('3+',counts>=3)]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    raw={r['source']:r for r in read(OUT/'run22/all_inputs_frozen.json')['sources']}
    labels={r['source']:r for r in read(OUT/'run78/training_targets.json')['rows']}
    manifest=read(OUT/'run26/block_manifest.json');rows=[]
    write(dest/'registered_protocol.json',dict(stage='TRAIN_RAW_MULTIPLICITY_AND_VOXEL_ALIAS_DIAGNOSIS',sources=TRAIN,
        no_training=True,no_development_reference_read=True,raw_duplicate_rule_diagnostic_not_material_identity=True,
        no_point_removal_or_relabeling=True,source_sha256=digest(__file__),start_utc=utc()))
    for source in TRAIN:
        check_resources();print('AMBIGUITY_INPUT_AUDIT_START',source,flush=True)
        b=raw[source];r=labels[source];case=manifest['cases'][str(source)]
        assert digest(OUT/'run22'/b['xyz_file'])==b['xyz_sha256'] and digest(OUT/'run78'/r['file'])==r['sha256']
        xyz=np.load(OUT/'run22'/b['xyz_file']);y=np.load(OUT/'run78'/r['file'])
        _,inverse,counts=np.unique(xyz,axis=0,return_inverse=True,return_counts=True);duplicates=counts[inverse]
        voxel_multiplicity=np.empty(len(xyz),np.int32);seen=np.zeros(len(xyz),np.uint8);alias_errors=conflict_core_points=0;voxel_count=0
        mapping=OUT/f'run26/blocks_{source:02d}/input_blocks.npz';assert digest(mapping)==case['mapping_sha256']
        with np.load(mapping) as trace:
            for bi,block_binding in enumerate(case['blocks']):
                path=OUT/'run26'/block_binding['path'];assert digest(path)==block_binding['sha256']
                ids=trace[f'ids_{bi}'].astype(np.int64);n=int(trace[f'core_count_{bi}']);core_ids=ids[:n]
                with np.load(path) as block:
                    inv=block['x_inverse'].astype(np.int64);nv=len(block['x_coord']);assert block['x_feat'].shape[1]==3
                    assert np.array_equal(block['x_feat'],block['x_coord'])
                    count=np.bincount(inv,minlength=nv);voxel_multiplicity[core_ids]=count[inv[:n]];seen[core_ids]+=1
                    target=y[core_ids];zero=np.bincount(inv[:n][target==0],minlength=nv);one=np.bincount(inv[:n][target==1],minlength=nv)
                    conflict=(zero>0)&(one>0);alias_errors+=int(np.minimum(zero,one).sum())
                    conflict_core_points+=int((zero[conflict]+one[conflict]).sum());voxel_count+=int((zero+one>0).sum())
        assert np.all(seen==1);known=int((y>=0).sum())
        row=dict(source=source,original_points=len(y),known_reference_points=known,
            exact_duplicate_counts=histogram(duplicates,y),voxel_multiplicity_counts=histogram(voxel_multiplicity,y),
            exact_duplicate_greater_one_diagnostic=scores(confusion((duplicates>1).astype(float),y)),
            voxel_count_greater_one_diagnostic=scores(confusion((voxel_multiplicity>1).astype(float),y)),
            conflicting_core_points_sharing_voxel=conflict_core_points,unavoidable_errors_if_one_prediction_per_core_voxel=alias_errors,
            optimistic_point_accuracy_upper_bound_for_voxel_alias_only=1-alias_errors/max(1,known),
            known_core_voxels=voxel_count,input_features_are_only_xyz=True,multiplicity_not_explicit_input=True)
        rows.append(row);write(dest/f'source_{source:02d}_input_audit.json',row)
        print('AMBIGUITY_INPUT_AUDIT_COMPLETE',source,'DUPLICATE_DIAGNOSTIC',row['exact_duplicate_greater_one_diagnostic'],
            'VOXEL_ALIAS_ERRORS',alias_errors,'/',known,flush=True)
    write(dest/'receipt.json',dict(status='TRAIN_AMBIGUITY_INPUT_DIAGNOSED',rows=rows,optimizer_updates=0,
        full_contract_G=496,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
