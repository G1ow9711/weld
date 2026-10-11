"""TRAIN-only sparse positive supervision for every legal endpoint pairing.

All lower-q to higher-q pairs remain implicitly present. Endpoint model397 is
not read: its poor DEV recall must not remove geometric possibilities.
"""
import argparse
import gc
import time
from collections import defaultdict
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_endpoint_role_train_1246 import inputs
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case


def prepare(dest,input_run='run396'):
    if (OUT/'run404/confirmed_supervision_conflicts.json').exists():
        raise RuntimeError('Inherited run380 whole-carrier mapping omits accepted native-D3 positives; use rebuilt native compatibility')
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for
    start=time.perf_counter();cases=[];all_classes={0};norm_sum=np.zeros(180);norm_sq=np.zeros(180);norm_n=0
    for key in TRAIN:
        x,m,ir=inputs(key,input_run);case=load_case(key);g=case['graph'];case['x']._mmap.close();target,tbind=target_for(key)
        folder=OUT/f'run380/source_{key}';sr=read(folder/'joint_supervision_receipt.json')
        assert digest(folder/'joint_count_targets.npz')==sr['labels_sha256']
        assert digest(folder/'joint_interval_instances.json')==sr['instances_sha256']
        with np.load(folder/'joint_count_targets.npz') as z:counts=z['labels']
        events=defaultdict(list);all_owners=set()
        for event in read(folder/'joint_interval_instances.json')['instances']:
            all_owners.add(tuple(event['owner']))
            if event['count_is_known']:events[event['row']].append(event)
        records=[];owner_mass=defaultdict(float);covered=set();conflicts=[];total_pairs=0
        for i,row in enumerate(m['rows']):
            row=int(row)
            if counts[row]<0:continue
            a,b=map(int,m['offsets'][i:i+2]);n=b-a;q=m['q_mm'][a:b];curve=g.curve(row//2);points=curve.at(q)
            values=x[2*a:2*b].astype(np.float64)
            norm_sum+=values.sum(0);norm_sq+=(values*values).sum(0);norm_n+=len(values)
            record=dict(source=key,row=row,q_offset=a,feature_offset=2*a,n=n,curve_length_mm=curve.length,
                        is_closed=bool(curve.kind=='arc' and curve.arc.is_closed),pair_count=n*(n-1)//2,events=[])
            total_pairs+=record['pair_count']
            for event in events[row]:
                owner=tuple(event['owner']);pi,ci=owner
                truth=np.asarray(target['parents'][pi]['children'][ci]['original_points_mm'])[[0,-1]]
                native=curve.at(np.asarray(event['ratio'])*event['length_mm'])
                if np.linalg.norm(native-truth[::-1],axis=1).sum()<np.linalg.norm(native-truth,axis=1).sum():truth=truth[::-1]
                starts=np.flatnonzero(np.linalg.norm(points-truth[0],axis=1)<=3)
                ends=np.flatnonzero(np.linalg.norm(points-truth[1],axis=1)<=3)
                pairs=np.array([int(si*n+ei) for si in starts for ei in ends if si<ei],np.int64)
                if not len(pairs):continue
                covered.add(owner);owner_mass[owner]+=event['weight'];all_classes.add(int(event['category']))
                record['events'].append(dict(owner=owner,label=int(event['category']),
                    weight=float(event['weight']),pairs=pairs))
            records.append(record)
        missing=all_owners-covered
        assert not missing,(key,missing)
        total_positive=0
        for record in records:
            positive={}
            for event in record.pop('events'):
                # Every original child has total positive mass 1/(2*3*G_case).
                weight=event['weight']/owner_mass[event['owner']]/len(event['pairs'])/(2*len(TRAIN)*len(all_owners))
                for index in event['pairs']:
                    index=int(index)
                    if index in positive and positive[index][0]!=event['label']:
                        conflicts.append(dict(row=record['row'],index=index,labels=[positive[index][0],event['label']]))
                    elif index in positive:positive[index][1]+=weight
                    else:positive[index]=[event['label'],weight]
            indices=sorted(positive)
            record['positive_indices']=indices
            record['positive_labels']=[positive[j][0] for j in indices]
            record['positive_weights']=[positive[j][1] for j in indices]
            total_positive+=len(indices)
        assert not conflicts,conflicts[:10]
        negative_count=total_pairs-total_positive
        negative_weight=1/(2*len(TRAIN)*negative_count) if negative_count else 0.
        positive_mass=sum(sum(r['positive_weights']) for r in records)
        assert abs(positive_mass-1/(2*len(TRAIN)))<1e-8
        output=dict(source=key,records=records,negative_weight=negative_weight,
            all_pairs=total_pairs,positive_pairs=total_positive,negative_pairs=negative_count,
            original_children=len(all_owners),covered_children=len(covered),original_owner_ids_are_supervision_only=True,
            feature_sha256=ir['feature_sha256'],map_sha256=ir['map_sha256'],target_sha256=tbind['target_sha256'],
            positive_mass=positive_mass,negative_mass=negative_count*negative_weight)
        path=dest/f'source_{key}_pair_targets.json';write(path,output)
        cases.append(dict(source=key,rows=len(records),pairs=total_pairs,positive_pairs=total_positive,
                          G=len(all_owners),covered=len(covered),targets_sha256=digest(path)))
        print('JOINT_PAIR_SUPERVISION',cases[-1],flush=True)
        x._mmap.close();del case,g,records,output;gc.collect()
    mean=norm_sum/norm_n;scale=np.sqrt(np.maximum(norm_sq/norm_n-mean*mean,0));scale[scale<1e-6]=1
    np.savez(dest/'normalization.npz',mean=mean.astype(np.float32),scale=scale.astype(np.float32))
    write(dest/'receipt.json',dict(status='ALL_ANCHOR_PAIR_TRAIN_SUPERVISION_READY',TRAIN=list(TRAIN),cases=cases,
        G=sum(c['G'] for c in cases),all_pairs=sum(c['pairs'] for c in cases),
        classes=sorted(all_classes),normalization_sha256=digest(dest/'normalization.npz'),normalization_rows=norm_n,
        endpoint_gate_used=False,model397_scores_read=False,DEV_reads=0,STEP_reads=0,fit_calls=0,
        full_contract_pass=False,seconds=time.perf_counter()-start))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--input-run',default='run396')
    ap.add_argument('--geometry-run',default='run395');args=ap.parse_args();dest=OUT/args.run
    dest.mkdir(exist_ok=False);write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),
        TRAIN=list(TRAIN),input_run=args.input_run,geometry_run=args.geometry_run,all_legal_pairs_implicit=True,
        positive_endpoint_tolerance_mm=3.,no_GT_forced_candidate_insertion=True,DEV_reads=0,STEP_reads=0,fit_calls=0))
    prepare(dest,args.input_run)


if __name__=='__main__':main()
