"""Verify ambiguity-head coverage and identical-relation gating comparison."""
import argparse
import json
import numpy as np
import torch
from weld_pipeline.sjg_prepare_776 import OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_material_ambiguity_874 import MaterialAmbiguityHead
from weld_pipeline.sjg_constrained_material_870 import verify_constraints
from weld_pipeline.sjg_context_compare_871 import overlap_detection
from weld_pipeline.sjg_material_atom_audit_873 import atom_union_bound


def controls():
    # The interior-only union diagnostic is not a bound for later boundary
    # attachment. Explicitly retain this distinction in the evidence.
    c=np.array([0,0,-1,-1]);owner=np.array([2,2,5,5]);r=atom_union_bound(c,owner)
    assert r['optimistic_matched_at_half']==1 and r['reference_owners']==2
    q=overlap_detection(np.arange(5),np.array([0,0,1,1]),owner)
    assert q['matched_reference_owners_at_IoU_half']==2
    return dict(interior_union_bound_excludes_later_boundary_attachment=True,known_boundary_points_remain_in_reference_denominator=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);write(dest/'controls.json',controls());training=OUT/'run79';data=OUT/'run78';test=OUT/'run80'
    receipt=read(training/'receipt.json');protocol=read(training/'registered_protocol.json');expected=read(data/'receipt.json')
    assert receipt['status']=='SEPARATE_MATERIAL_AMBIGUITY_HEAD_TRAINED' and receipt['completed_full_epochs']==8 and receipt['backbone_updates']==0
    epochs=[json.loads(x) for x in (training/'epochs.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in epochs]==list(range(1,9)) and receipt['actual_optimizer_updates']==protocol['expected_total_updates']
    for r in epochs:
        assert r['known_points']==expected['positive']+expected['negative'] and r['positive']==expected['positive'] and r['negative']==expected['negative']
        assert r['unknown_retained_outside_loss']==expected['unknown']
    assert digest(training/'last.pt')==receipt['checkpoint_sha256']
    assert digest(OUT/'run74/dense_control/last.pt')==protocol['pretrained_backbone_sha256']
    assert read(test/'registered_protocol.json')['checkpoint_sha256']==receipt['checkpoint_sha256']
    assert read(test/'registered_protocol.json')['frozen_backbone_sha256']==protocol['pretrained_backbone_sha256']
    with (training/'last.pt').open('rb') as h:state=torch.load(h,map_location='cpu')
    assert not any(k.startswith('backbone.') for k in state['model'])
    model=MaterialAmbiguityHead(np.zeros(32),np.ones(32));model.load_state_dict(state['model']);model.eval();del state
    features={r['source']:r for r in read(data/'all_inputs_frozen.json')['sources']}
    original={r['source']:r for r in read(OUT/'run22/all_inputs_frozen.json')['sources']}
    base=OUT/'run75/dense_control';relations={r['source']:r for r in read(base/'all_predictions_frozen.json')['sources']}
    structures={r['source']:r for r in read(base/'constrained_predictions_frozen.json')['sources']}
    bindings=read(test/'all_predictions_frozen.json')['sources'];assert [r['source'] for r in bindings]==TRAIN+VALID+ARC
    total=0;checks=[]
    for b in bindings:
        check_resources();source=b['source'];n=b['points'];total+=n;print('AMBIGUITY_AUDIT_START',source,n,flush=True)
        r=relations[source];f=features[source];s=structures[source]
        assert n==original[source]['points'] and b['raw_sha256']==original[source]['raw_sha256']==f['raw_sha256']
        assert digest(test/b['file'])==b['sha256'] and digest(data/f['file'])==b['feature_sha256']==f['sha256']
        assert digest(base/r['file'])==b['relation_sha256']==r['sha256']
        assert digest(base/s['file'])==b['baseline_structure_sha256']==s['sha256']
        assert f['every_original_point_one_core'] and f['normal_max_abs_difference']<1e-4 and f['boundary_max_abs_difference']<1e-4
        x=np.load(data/f['file'],mmap_mode='r');assert x.shape==(n,32)
        with np.load(test/b['file']) as q,np.load(base/r['file']) as p:
            prob=q['material_ambiguity_probability'];assert len(prob)==n and np.isfinite(prob).all()
            assert np.all((prob>=0)&(prob<=1)) and np.array_equal(q['boundary_uncertain'].astype(bool),prob>=.5)
            constraints=verify_constraints(p['neighbors'],p['material_distance'],q)
            assert constraints==b['stats']['observed_negative_edges']
            counts=np.diff(q['offsets']);assert int((counts==0).sum())==b['stats']['unknown_membership_points']
            assert int((counts>1).sum())==b['stats']['multi_membership_points']
            ids=np.unique(np.linspace(0,n-1,min(4096,n)).astype(np.int64))
            with torch.inference_mode():replayed=model(torch.from_numpy(np.array(x[ids]))).sigmoid().numpy()
            error=float(np.abs(replayed-prob[ids]).max());assert error<1e-5
        checks.append(dict(source=source,points=n,verified_negative_constraints=constraints,CPU_head_replay_points=len(ids),CPU_max_abs_difference=error))
        print('AMBIGUITY_AUDIT_COMPLETE',checks[-1],flush=True)
    assert total==2641757;results=read(test/'receipt.json');assert results['status']=='SEPARATE_AMBIGUITY_GATING_COMPARED'
    write(dest/'receipt.json',dict(status='MATERIAL_AMBIGUITY_COMPARISON_VERIFIED',rows=results['rows'],checks=checks,
        all8_points=total,training_optimizer_updates=receipt['actual_optimizer_updates'],backbone_updates=0,
        all_input_checkpoint_and_output_hashes_verified=True,all_known_TRAIN_points_in_each_epoch=True,
        all_original_point_denominators_preserved=True,all_negative_constraints_verified=True,
        cached_feature_replay_not_end_to_end_timing=True,weld_extraction_not_rerun=True,full_contract_G=496,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
