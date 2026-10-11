"""Audit accepted-family distance targets and frozen internal split before fit."""
from pathlib import Path
from collections import Counter
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    source=BASE/'reference_distance_fields_v1'
    receipt=read(source/'receipt.json')
    split_path=BASE/'reference_distance_auxiliary_protocol_v1/split.json'
    split=read(split_path)
    protocol=read(source/'protocol.json')
    assert protocol['label_rebuild_summary_sha256']==digest(BASE/'training_label_rebuild_summary.json')
    assert protocol['group_manifest_sha256']==split['group_manifest_sha256']==digest(BASE/'group_manifest.json')
    cases={r['key']:r for r in split['cases']}
    groups={r['key']:r for r in read(BASE/'group_manifest.json')['groups']}
    assert len(cases)==len(receipt['cases'])==58
    assert Counter(r['role'] for r in cases.values())=={'train':47,'development_validation':11}
    assert receipt['protocol_sha256']==digest(source/'protocol.json')
    assert set(cases)=={r['key'] for r in receipt['cases']}
    assert len({r['family_id'] for r in cases.values()})==58
    rows=[]
    for r in receipt['cases']:
        key=r['key'];case=cases[key];folder=source/key
        assert read(folder/'receipt.json')==r
        assert r['family_id']==case['family_id']==groups[key]['family_id']
        assert groups[key]['role']=='EXPANSION_TRAIN'
        assert r['input_receipt_sha256']==case['input_receipt_sha256']==digest(BASE/'inputs'/key/'receipt.json')
        assert r['label_receipt_sha256']==case['label_receipt_sha256']==digest(BASE/'labels'/key/'receipt.json')
        assert r['target_sha256']==case['target_sha256']==digest(BASE/'labels'/key/'targets.json')
        assert r['input_sha256']==digest(BASE/'inputs'/key/'xyz_mm.npy')
        for name in ('lower','upper'):
            assert digest(folder/f'{name}_mm.npy')==r[f'{name}_sha256']
        lo=np.load(folder/'lower_mm.npy',mmap_mode='r');hi=np.load(folder/'upper_mm.npy',mmap_mode='r')
        assert lo.shape==hi.shape==(r['points'],) and lo.dtype==hi.dtype==np.float32
        near=far=0
        for a in range(0,len(lo),65536):
            low=lo[a:a+65536];high=hi[a:a+65536]
            assert np.isfinite(low).all() and np.isfinite(high).all()
            assert ((low>=0)&(high<=100)&(low<=high)).all()
            assert (high-low<=.25003).all()
            near+=int((high<=3).sum());far+=int((low>=6).sum())
        assert near==r['certain_reference_distance_le3mm'] and far==r['certain_reference_distance_ge6mm']
        rows.append(dict(key=key,role=case['role'],points=len(lo),near_points=near,near_fraction=near/len(lo),far_points=far,reference_children=r['reference_children']))
    out=BASE/'reference_distance_fields_audit_v1';out.mkdir(exist_ok=False)
    write(out/'report.json',dict(status='PASS',field_receipt_sha256=digest(source/'receipt.json'),split_sha256=digest(split_path),script_sha256=digest(__file__),role_counts=dict(Counter(r['role'] for r in rows)),total_points=sum(r['points'] for r in rows),cases_without_near_points=[r['key'] for r in rows if not r['near_points']],cases=rows,scope='Integrity, target bounds, accepted family split and reference proximity prevalence. No trained accuracy, physical negatives or weld extraction proof.',protected_DEV_reads=0))
    print('PASS',len(rows),'families',sum(r['points'] for r in rows),'points',flush=True)

if __name__=='__main__':main()
