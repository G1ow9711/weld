"""Independent receipt, point-denominator and signed overlap constraint audit."""
import argparse
import json
import numpy as np
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_constrained_material_870 import verify_constraints,constrained_memberships
from weld_pipeline.sjg_context_compare_871 import ARMS,overlap_detection


def controls():
    rng=np.random.default_rng(872)
    for _ in range(40):
        n=30;neighbors=rng.integers(0,n,size=(n,5),dtype=np.int32)
        neighbors[neighbors==np.arange(n)[:,None]]=-1
        distance=rng.random((n,5)).astype(np.float32);distance[neighbors<0]=np.nan
        out,_=constrained_memberships(neighbors,distance,rng.random(n));verify_constraints(neighbors,distance,out)
    owner=np.array([1,1,2,2,-1]);off=np.arange(6);mem=np.array([0,0,1,1,1])
    perfect=overlap_detection(off,mem,owner);assert perfect['matched_reference_owners_at_IoU_half']==2
    missing=overlap_detection(np.array([0,1,1,1,1,1]),np.array([0]),owner)
    assert missing['reference_owners']==2 and missing['known_unassigned_points']==3 and missing['missed_reference_owners_at_IoU_half']==1
    mixed=overlap_detection(off,np.zeros(5,np.int32),owner)
    assert mixed['matched_reference_owners_at_IoU_half']==1
    return dict(random_signed_graph_checks=40,known_unassigned_stays_in_truth_denominator=True,
        unknown_reference_separate=True,one_prediction_cannot_match_two_owners=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);write(dest/'controls.json',controls())
    assert read(OUT/'run75/receipt.json')['status']=='CONTEXT_MATERIAL_COMPARISON_COMPLETE'
    negative_data=read(OUT/'run73/receipt.json');initial=digest(OUT/'run70/dense_material/last.pt')
    for arm in ARMS:
        folder=OUT/'run74'/arm;receipt=read(folder/'receipt.json');protocol=read(folder/'registered_protocol.json')
        epochs=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
        assert receipt['completed_full_epochs']==2 and receipt['actual_optimizer_updates']==1168
        assert receipt['checkpoint_sha256']==digest(folder/'last.pt') and protocol['initial_checkpoint_sha256']==initial
        assert len(epochs)==2 and [r['epoch'] for r in epochs]==[1,2]
        for r in epochs:
            assert r['core_points']==1880869 and r['known_material_points']==r['directly_supervised_material_points']==1747537
            assert r['local_negative_pairs']==(negative_data['negative_pairs_per_epoch'] if arm=='context_negative' else 0)
            assert r['local_negative_core_anchors']==(negative_data['unique_core_anchors_per_epoch'] if arm=='context_negative' else 0)
    for binding in read(OUT/'run75/all_arms_frozen_before_reference.json')['arms']:
        folder=OUT/'run75'/binding['arm']
        assert digest(folder/'all_predictions_frozen.json')==binding['relation_manifest_sha256']
        assert digest(folder/'constrained_predictions_frozen.json')==binding['structure_manifest_sha256']
    inputs={r['source']:r for r in read(OUT/'run22/all_inputs_frozen.json')['sources']};summary={};checks=[]
    for arm in ARMS:
        folder=OUT/'run75'/arm;rows=[];total=0
        assert read(folder/'registered_protocol.json')['checkpoint_sha256']==read(OUT/'run74'/arm/'receipt.json')['checkpoint_sha256']
        metrics={r['source']:r for r in read(folder/'offline_material_metrics.json')}
        structures={r['source']:r for r in read(folder/'structure_metrics.json')}
        bindings=read(folder/'all_predictions_frozen.json')['sources']
        assert [r['source'] for r in bindings]==TRAIN+VALID+ARC
        cb={r['source']:r for r in read(folder/'constrained_predictions_frozen.json')['sources']}
        for r in bindings:
            check_resources();source=r['source'];n=r['points'];total+=n
            print('CONTEXT_AUDIT_START',arm,source,'ORIGINAL_POINTS',n,flush=True)
            assert r['raw_sha256']==inputs[source]['raw_sha256'] and n==inputs[source]['points']
            assert digest(folder/r['file'])==r['sha256'] and digest(folder/cb[source]['file'])==cb[source]['sha256']
            assert cb[source]['relation_sha256']==r['sha256'] and cb[source]['points']==n
            assert r['python_STEP_or_teacher_reads']==0 and r['python_file_opens_during_numeric_inference']==[]
            with np.load(folder/r['file']) as p,np.load(folder/cb[source]['file']) as q:
                near=p['neighbors'];distance=p['material_distance'];assert near.shape==distance.shape==(n,8)
                assert np.all((near>=-1)&(near<n)) and not np.any(near==np.arange(n)[:,None])
                assert np.all(p['core_coverage']==1) and np.all(p['core_block']>=0)
                assert not np.any(np.isfinite(distance)&(near<0)) and np.isfinite(p['boundary']).all()
                constraints=verify_constraints(near,distance,q)
                assert constraints==cb[source]['stats']['observed_negative_edges']
                assert np.all(q['membership']>=0) and np.all(q['membership']<cb[source]['stats']['hypotheses'])
                assert len(q['interior_component'])==len(q['boundary_uncertain'])==n
                counts=np.diff(q['offsets']);assert int((counts==0).sum())==cb[source]['stats']['unknown_membership_points']
                assert int((counts>1).sum())==cb[source]['stats']['multi_membership_points']
            checks.append(dict(arm=arm,source=source,original_points=n,verified_negative_constraints=constraints))
            if source==24:rows.append(dict(source=source,status='NO_OFFLINE_STRUCTURE_REFERENCE',original_points=n));continue
            m=metrics[source];relation=m['relations']['material_distance']['all'];b=m['boundary'];s=structures[source]
            rows.append(dict(source=source,original_points=n,material_relations=relation,
                balanced_relation_recall=.5*(relation['same_recall']+relation['different_recall']),
                cross_surface_same_owner_recall=m['relations']['material_distance']['different_surfaces_same_owner']['same_recall'],
                normal_p95_deg=m['unoriented_normal_p95_deg'],boundary_F1=2*b['TP']/max(1,2*b['TP']+b['FP']+b['FN']),
                positive_connected_components=s['positive_connected_components'],constrained_multi_membership=s['constrained_multi_membership'],decoder_stats=s['decoder_stats']))
            print('CONTEXT_AUDIT_SOURCE',arm,source,'NEGATIVE_CONSTRAINTS',constraints,flush=True)
        assert total==2641757;summary[arm]=rows
    write(dest/'receipt.json',dict(status='CONTEXT_MATERIAL_COMPARISON_VERIFIED',summary=summary,checks=checks,
        actual_training_optimizer_updates=2336,all_raw_checkpoint_prediction_hashes_match=True,all_original_points_preserved=True,
        all_observed_negative_constraints_verified=True,full_contract_G=496,weld_extraction_not_rerun=True,
        full_contract_pass=False,source_sha256=digest(__file__),end_utc=utc()))


if __name__=='__main__':main()
