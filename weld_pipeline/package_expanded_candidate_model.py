"""Add expanded plate weights without changing any geometric model tensor."""
import json
import shutil
from pathlib import Path

import joblib
import torch

from training.labels import sha256


def main():
    parent=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v2')
    attrs=Path('weld_pipeline/runs/expanded_plate_attributes_seed132')
    if not (attrs/'complete.json').exists():
        raise ValueError('Expanded training unfinished')
    out=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v3');out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():
        raise ValueError('Frozen expanded candidate exists')
    state=torch.load(parent/'best.pt',map_location='cpu')
    geometry={k:v.clone() for k,v in state['horizontal']['model'].items()}
    state['attribute_heads']=joblib.load(attrs/'heads.joblib')
    state['source_hashes'].update(state['attribute_heads']['config']['sources'])
    state['source_hashes'].update({p:sha256(p) for p in ['weld_pipeline/predict_expanded_candidate_model.py',
        'weld_pipeline/package_expanded_candidate_model.py']})
    state.update(output_schema_version='expanded_candidate_attributes_v3',
        attribute_scope='Expanded unreviewed STEP-derived FIT candidates. Untrained-reference comparison available; E2 and complete contract not achieved.')
    if any(not torch.equal(v,state['horizontal']['model'][k]) for k,v in geometry.items()):
        raise ValueError('Geometry changed during attribute packaging')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for name,wanted in {**state['horizontal']['source_hashes'],**state['source_hashes']}.items():
        source=Path(name)
        if not source.exists() or sha256(source)!=wanted:
            source=parent/'source_snapshot'/source.name
        if sha256(source)!=wanted:
            raise ValueError('Missing source evidence: '+name)
        shutil.copy2(source,archive/source.name)
    torch.save(state,out/'best.pt')
    receipt=json.loads((parent/'training_receipt.json').read_text('utf8'))
    receipt.update(checkpoint_sha256=sha256(out/'best.pt'),parent_checkpoint_sha256=sha256(parent/'best.pt'),
        output_schema_version='expanded_candidate_attributes_v3',geometry_weights_changed=False,
        attribute_weights_changed=True,attribute_candidate_fit=json.loads((attrs/'complete.json').read_text('utf8')),
        raw_verified=False,full_contract_pass=False)
    (out/'training_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print('EXPANDED_CANDIDATE_PACKAGED',receipt['checkpoint_sha256'],flush=True)


if __name__=='__main__':
    main()
