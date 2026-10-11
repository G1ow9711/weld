"""Native interval compatibility -> occupancy targets with uncertain boundaries.

This is an auxiliary target set, not weld extraction or contract acceptance.
No DEV labels are read. Every original TRAIN child stays in the owner registry,
including short children without a guaranteed interior sequence sample.
"""
import gc
import shutil
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources

TRAIN=('21','BM79A_native_local','10','25','36')


def occupancy_labels(grid, windows):
    labels=np.zeros(len(grid),np.int8)
    possible=np.zeros(len(grid),bool);certain=np.zeros(len(grid),bool)
    for start_min,start_max,end_min,end_max in windows:
        assert start_min<=start_max<end_min<=end_max
        possible |= (grid>=start_min)&(grid<=end_max)
        certain |= (grid>=start_max)&(grid<=end_min)
    labels[possible]=-1;labels[certain]=1
    return labels


def controls():
    q=np.arange(11.)
    assert occupancy_labels(q,[(1,3,7,9)]).tolist()==[0,-1,-1,1,1,1,1,1,-1,-1,0]
    assert np.array_equal(occupancy_labels(q,[(1,3,7,9)]),occupancy_labels(10-q,[(1,3,7,9)]))
    assert np.array_equal(occupancy_labels(q,[]),np.zeros(11,np.int8))
    # An uncertain boundary of one child cannot erase another child's certain interior.
    joined=occupancy_labels(q,[(0,0,4,6),(4,6,10,10)])
    assert joined[4]==joined[6]==1 and joined[5]==-1
    return dict(uncertain_boundaries_not_false_background=True,empty_known_carrier_is_negative=True,
                overlapping_certain_interior_wins=True,parameter_reversal=True)


def prepare():
    check_resources();dest=OUT/'run428';dest.mkdir(exist_ok=False)
    snapshot=dest/'source_snapshot';snapshot.mkdir();shutil.copy2(__file__,snapshot/'sjg_native_occupancy_targets_1270.py')
    audit=read(OUT/'run422/independent_audit_summary.json')
    assert audit['status']=='PASS' and audit['original_G']==269
    source=read(OUT/'run423/receipt.json');assert source['G']==269
    assert {c['source'] for c in source['cases']}==set(TRAIN)
    write(dest/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=[],G=269,source_sha256=digest(__file__),
        native_interval_targets='run423',sequence_input='run421/sequence56',
        definition='positive inside every compatible endpoint pair; unknown in boundary tolerance envelope; negative outside all envelopes on known complete rows',
        unknown_background_rows_excluded=True,GT_not_used_to_generate_input_features=True,fit_calls=0,
        original_child_scope_not_changed=True,full_contract_pass=False))
    write(dest/'controls.json',controls());cases=[]
    for case in source['cases']:
        key=case['source'];check_resources()
        path=OUT/f'run423/source_{key}_set_targets.json';assert digest(path)==case['target_sha256']
        data=read(path);ep=OUT/f'run421/endpoints/source_{key}'
        assert digest(ep/'role_map.npz')==data['map_sha256']
        native_receipt=read(OUT/f'run422/source_{key}/receipt.json')
        assert native_receipt['compatibility_sha256']==data['native_compatibility_sha256']
        assert native_receipt['pairs_sha256']==data['native_pairs_sha256']
        assert digest(OUT/f'run422/source_{key}/compatibility.json')==data['native_compatibility_sha256']
        with np.load(ep/'role_map.npz') as z:endpoint_q=z['q_mm']
        seq=OUT/f'run421/sequence56/source_{key}';sr=read(seq/'receipt.json')
        assert digest(seq/'sequence_map.npz')==sr['map_sha256']
        assert sr['GT_reads']==0
        with np.load(seq/'sequence_map.npz') as z:m={k:z[k] for k in z.files}
        lookup={int(p):i for i,p in enumerate(m['profile_ids'])};admitted=set(map(int,m['admitted_rows']))
        rows=[];blocks=[];offsets=[0];owner_registry={};windows_records=[]
        for r in data['records']:
            row=r['row'];assert row in admitted
            pid_index=lookup[row//2];a,b=map(int,m['offsets'][pid_index:pid_index+2]);grid=m['q_mm'][a:b]
            n=r['n'];epq=endpoint_q[r['q_offset']:r['q_offset']+n];windows=[]
            for group in r['gold_groups']:
                pairs=np.asarray(group['pairs'],np.int64);left=epq[pairs//n];right=epq[pairs%n]
                window=(float(left.min()),float(left.max()),float(right.min()),float(right.max()))
                windows.append(window);owner=tuple(group['owner'])
                entry=owner_registry.setdefault(owner,dict(owner=list(owner),known_carriers=0,certain_positive_sites=0))
                entry['known_carriers']+=1
                entry['certain_positive_sites']+=int(((grid>=window[1])&(grid<=window[2])).sum())
            y=occupancy_labels(grid,windows);blocks.append(y);rows.append(row);offsets.append(offsets[-1]+len(y))
            windows_records.append(dict(row=row,windows=windows,owners=[v['owner'] for v in r['gold_groups']]))
        assert len(owner_registry)==data['original_G']
        labels=np.concatenate(blocks);folder=dest/f'source_{key}';folder.mkdir()
        np.savez_compressed(folder/'occupancy_targets.npz',rows=np.asarray(rows,np.int64),
            offsets=np.asarray(offsets,np.int64),labels=labels)
        write(folder/'owner_registry.json',dict(original_G=data['original_G'],owners=list(owner_registry.values()),windows=windows_records))
        result=dict(source=key,G=data['original_G'],registered_original_children=len(owner_registry),known_sequences=len(rows),
            positive_sites=int((labels==1).sum()),negative_sites=int((labels==0).sum()),uncertain_boundary_sites=int((labels==-1).sum()),
            children_without_certain_interior_sites=[v['owner'] for v in owner_registry.values() if not v['certain_positive_sites']],
            labels_sha256=digest(folder/'occupancy_targets.npz'),owner_registry_sha256=digest(folder/'owner_registry.json'),
            native_set_targets_sha256=case['target_sha256'],sequence_map_sha256=sr['map_sha256'],
            sequence_features_sha256=sr['features_sha256'],original_target_sha256=data['original_target_sha256'],
            all_original_children_retained=True,DEV_reads=0,fit_calls=0,full_contract_pass=False)
        write(folder/'receipt.json',result);cases.append(result)
        print('NATIVE_OCCUPANCY_TARGETS',key,'G',result['G'],'POS',result['positive_sites'],'NEG',result['negative_sites'],
              'UNCERTAIN',result['uncertain_boundary_sites'],flush=True)
        del data,blocks,labels,m,endpoint_q;gc.collect()
    assert sum(v['G'] for v in cases)==269
    write(dest/'receipt.json',dict(status='NATIVE_OCCUPANCY_TARGETS_PENDING_INDEPENDENT_AUDIT',cases=cases,G=269,
        fit_calls=0,DEV_reads=0,full_contract_pass=False))


if __name__=='__main__':prepare()
