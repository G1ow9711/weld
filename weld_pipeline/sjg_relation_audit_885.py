"""Validate learned relation ablation, full-point output and deterministic decoder replay."""
import argparse
import json
import numpy as np
import torch
from scipy.sparse import csr_matrix
from weld_pipeline.sjg_prepare_776 import OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_relation_features_880 import RelationHead,FEATURE_NAMES,pair_features
from weld_pipeline.sjg_train_relation_882 import ARMS
from weld_pipeline.sjg_soft_material_883 import decode_soft,merge_signed_graph,controls


def feature_controls():
    rng=np.random.default_rng(885);xyz=rng.normal(size=(10,3));near=np.empty((10,8),np.int32)
    for i in range(10):near[i]=np.array([j for j in range(10) if j!=i])[:8]
    normal=rng.normal(size=(10,3));normal/=np.linalg.norm(normal,axis=1)[:,None]
    material=rng.random((10,8));surface=rng.random((10,8));edge=rng.random(10);amb=rng.random(10)
    support=csr_matrix((rng.random((10,4))>.5).astype(np.int32));blocks=np.arange(10)//3
    x,flat=pair_features(xyz,near,material,surface,normal,edge,amb,support,blocks)
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    rotated,idx=pair_features(xyz@q+np.array([50.,20.,-100.]),near,material,surface,normal@q,edge,amb,support,blocks)
    assert np.array_equal(flat,idx) and np.allclose(x,rotated,atol=1e-5,rtol=1e-5)
    model=RelationHead(np.zeros(20),np.ones(20),False);changed=x.copy();changed[:,2:]+=100
    with torch.inference_mode():assert torch.equal(model(torch.from_numpy(x)),model(torch.from_numpy(changed)))
    for _ in range(30):
        n=20;left,right=np.triu_indices(n,1);cost=rng.normal(size=len(left))
        merge_signed_graph(n,left,right,cost,np.ones(n,np.int64))
    return dict(rigid_transform_feature_invariance=True,distance_only_ablation_ignores_other_features=True,
        random_signed_graph_gain_checks=30,**controls())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    write(dest/'controls.json',feature_controls());data=OUT/'run83';test=OUT/'run85';training=OUT/'run84';base=OUT/'run75/dense_control'
    results=read(test/'receipt.json');assert results['status']=='PAIR_RELATION_SOFT_GROUPING_COMPARED'
    expected=read(data/'receipt.json');p=read(training/'registered_protocol.json')
    for arm in ARMS:
        folder=training/arm;receipt=read(folder/'receipt.json');epochs=[json.loads(x) for x in (folder/'epochs.jsonl').read_text().splitlines()]
        assert len(epochs)==4 and [r['epoch'] for r in epochs]==[1,2,3,4]
        assert receipt['actual_optimizer_updates']==p['updates_each'] and receipt['checkpoint_sha256']==digest(folder/'last.pt')
        for r in epochs:
            assert r['known_pairs']==expected['known_pairs'] and r['same']==expected['same'] and r['different']==expected['different']
    inputs={r['source']:r for r in read(data/'all_inputs_frozen.json')['sources']};original={r['source']:r for r in read(OUT/'run22/all_inputs_frozen.json')['sources']}
    relations={r['source']:r for r in read(base/'all_predictions_frozen.json')['sources']};seeds={r['source']:r for r in read(base/'constrained_predictions_frozen.json')['sources']}
    for b in read(test/'all_arms_frozen_before_reference.json')['arms']:assert digest(test/b['arm']/'all_predictions_frozen.json')==b['sha256']
    checks=[]
    for arm in ARMS:
        model=RelationHead(np.zeros(20),np.ones(20),arm=='finite_geometry')
        with (training/arm/'last.pt').open('rb') as h:state=torch.load(h,map_location='cpu')
        model.load_state_dict(state['model']);model.eval();del state
        assert torch.all(model.mask[:2]==1) and torch.all(model.mask[2:]==float(arm=='finite_geometry'))
        bindings=read(test/arm/'all_predictions_frozen.json')['sources'];assert [b['source'] for b in bindings]==TRAIN+VALID+ARC;total=0
        for b in bindings:
            check_resources();source=b['source'];n=b['points'];total+=n;print('SOFT_RELATION_AUDIT_START',arm,source,flush=True)
            i=inputs[source];r=relations[source];s=seeds[source]
            assert n==original[source]['points'] and b['raw_sha256']==original[source]['raw_sha256']
            assert digest(test/arm/b['file'])==b['sha256'] and digest(data/i['features_file'])==i['features_sha256']==b['features_sha256']
            assert digest(data/i['flat_file'])==i['flat_sha256']==b['flat_sha256'] and digest(base/r['file'])==b['relation_sha256']
            assert digest(base/s['file'])==b['seed_sha256'] and b['checkpoint_sha256']==digest(training/arm/'last.pt')
            flat=np.load(data/i['flat_file']);x=np.load(data/i['features_file'],mmap_mode='r')
            assert x.shape==(len(flat),len(FEATURE_NAMES)) and np.all(np.diff(flat)>0) and np.all((flat>=0)&(flat<n*8))
            with np.load(test/arm/b['file']) as q,np.load(base/r['file']) as rel,np.load(base/s['file']) as seed:
                assert len(q['offsets'])==n+1 and q['offsets'][0]==0 and q['offsets'][-1]==len(q['membership'])
                assert np.all(np.diff(q['offsets'])>=0) and np.all(q['membership']>=0)
                assert len(q['relation_logit'])==len(flat) and np.isfinite(q['relation_logit']).all()
                replay,stats=decode_soft(rel['neighbors'],flat,q['relation_logit'],seed['offsets'],seed['membership'])
                for name,value in replay.items():assert np.array_equal(value,q[name])
                for name,value in stats.items():assert value==b['stats'][name]
                ids=np.unique(np.linspace(0,len(x)-1,min(4096,len(x))).astype(np.int64))
                with torch.inference_mode():prediction=model(torch.from_numpy(np.array(x[ids]))).numpy()
                error=float(np.abs(prediction-q['relation_logit'][ids]).max());assert error<1e-4
                assert int((np.diff(q['offsets'])==0).sum())==stats['unknown_membership_points']
            checks.append(dict(arm=arm,source=source,points=n,full_decoder_replay_exact=True,CPU_head_sample_count=len(ids),CPU_logit_max_abs_error=error))
            print('SOFT_RELATION_AUDIT_COMPLETE',checks[-1],flush=True)
        assert total==2641757
    write(dest/'receipt.json',dict(status='PAIR_RELATION_SOFT_GROUPING_VERIFIED',rows=results['rows'],checks=checks,
        all_original_points_and_input_output_hashes_verified=True,full_decoder_replay_exact=True,
        training_optimizer_updates=2*p['updates_each'],all_known_train_pairs_each_epoch=True,
        full_contract_G=496,weld_extraction_not_rerun=True,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
