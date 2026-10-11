"""Prepare PCD-only material features; CAD classes and welding labels stay separate."""
import collections
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.material_support_ablation import side_sorted
from weld_pipeline.signed_material_support import SignedMaterialSupport, VERSION, FEATURE_COUNT, PARAMETERS

OUT = Path('weld_pipeline/data/material_support_v1')
AUDIT = Path('weld_pipeline/reports/candidate_material_identity_full_148v3_20261001')
CLASSES = ('FINITE_SIDE_MISSING', 'SAME_BODY_ONLY', 'CROSS_BODY_AVAILABLE_POSSIBLY_AMBIGUOUS')


def material_class(state):
    if state in ('BOTH_FINITE_SIDES_MISSING', 'ONE_FINITE_SIDE_MISSING'):
        return 0
    if state == 'SAME_BODY_ONLY':
        return 1
    if state in ('CROSS_BODY_ONLY','SAME_OR_CROSS_BODY_AMBIGUOUS'):
        return 2
    raise ValueError('Unknown offline material reference state')


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed material feature data')
    started = time.perf_counter()
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    audit_complete = json.loads((AUDIT/'complete.json').read_text('utf8'))
    physical_path = Path('weld_pipeline/reports/signed_material_support_physical_20261001.json')
    physical = json.loads(physical_path.read_text('utf8'))
    if not physical['all_passed'] or physical['extractor_sha256'] != sha256('weld_pipeline/signed_material_support.py'):
        raise ValueError('Current physical feature verification required')
    names = audit_complete['config']['pieces']
    if len(names)!=19 or any(n not in split['train'] or n in split['validation'] for n in names):
        raise ValueError('Exactly 19 registered CAD FIT sources required')
    folds_path = Path('weld_pipeline/reports/source_family_audit_20261001/summary.json')
    folds = json.loads(folds_path.read_text('utf8'))['folds']
    sources = [__file__, 'weld_pipeline/signed_material_support.py','weld_pipeline/material_support_ablation.py',
               'weld_pipeline/curved_cache.py','training/dataio.py']
    OUT.mkdir(parents=True)
    config = dict(version=VERSION,parameters=PARAMETERS,feature_count=FEATURE_COUNT,pieces=names,
        classes=list(CLASSES),folds=folds,folds_sha256=sha256(folds_path),split_sha256=sha256(split_path),
        audit_complete_sha256=sha256(AUDIT/'complete.json'),physical_sha256=sha256(physical_path),
        sources={p:sha256(p) for p in sources},source_inputs_label_free=True,
        material_reference_reviewed=False,sampling='Original fixed five candidate fractions, no truth-driven sampling',
        original_business_labels_unchanged=True,observed_check_read=False,full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for source in sources:shutil.copy2(source,snapshot/Path(source).name)
    manifests={};totals=collections.Counter()
    for name in names:
        root=OUT/name;root.mkdir()
        piece=load_piece(split['piece_roots'][name],verify=True)
        audit_path=AUDIT/(name+'.json');audit=json.loads(audit_path.read_text('utf8'))
        if audit['source_cache_manifest_sha256']!=piece['cache_manifest_sha256']:
            raise ValueError('Original candidate cache changed')
        pcd=Path(piece['record']['input'])
        if sha256(pcd)!=audit['input_sha256']:
            raise ValueError('PCD differs from material audit')
        extractor=SignedMaterialSupport(read_xyz_pcd(pcd)*1000.)
        features=[];material=[];business=[];entries=[]
        for candidate in audit['candidates']:
            si=candidate['sequence'];seq=piece['sequences'][si];target=piece['targets'][si]
            for sample in candidate['samples']:
                i=sample['sample_index']
                if not np.array_equal(seq['xyz'][i].astype(float),sample['xyz_mm']):
                    raise ValueError('Audit query differs from point candidate')
                raw=extractor.describe_sequence(seq,[i])[0]
                state=sample['material_hypothesis']['state']
                material.append(material_class(state))
                fg=bool(target[i,0]>.5) if target[i,0]>=0 else None
                if fg!=sample['reference_foreground_diagnostic_only']:
                    raise ValueError('Original business label changed')
                business.append(int(fg) if fg is not None else -1)
                features.append(raw)
                entries.append(dict(sequence=si,sample_index=i,type=seq['type'],offline_reference_state=state,
                                    material_ambiguity_is_retained=(state=='SAME_OR_CROSS_BODY_AMBIGUOUS')))
        x=np.asarray(features,np.float32)
        np.save(root/'signed.npy',x)
        np.save(root/'side_sorted.npy',side_sorted(x))
        np.savez(root/'labels.npz',material=np.asarray(material,np.int64),business=np.asarray(business,np.int64))
        manifest=dict(piece=name,entries=entries,rows=len(x),pcd_sha256=sha256(pcd),audit_sha256=sha256(audit_path),
            cache_manifest_sha256=piece['cache_manifest_sha256'],files={p:sha256(root/p) for p in ['signed.npy','side_sorted.npy','labels.npz']},
            material_counts=np.bincount(material,minlength=3).tolist(),business_known=int((np.array(business)>=0).sum()),
            business_foreground=int((np.array(business)==1).sum()),elapsed_seconds=time.perf_counter()-started)
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
        manifests[name]=sha256(root/'manifest.json')
        totals.update(dict(rows=len(x),business_foreground=manifest['business_foreground']))
        print('MATERIAL_PCD_FEATURES',name,len(x),'CLASSES',manifest['material_counts'],
              'FG',manifest['business_foreground'],'SECONDS',round(time.perf_counter()-started,2),flush=True)
        del piece,extractor,features,x
    complete=dict(config_sha256=sha256(OUT/'config.json'),manifests=manifests,totals=dict(totals),
                  elapsed_seconds=time.perf_counter()-started,observed_check_read=False,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')
    print('MATERIAL_PCD_FEATURES_COMPLETE',dict(totals),flush=True)


if __name__=='__main__':
    main()
