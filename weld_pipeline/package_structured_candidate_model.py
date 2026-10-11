"""Preserve geometric weights while packaging structural thickness candidates."""
import json
import shutil
from pathlib import Path

import joblib
import torch

from training.labels import sha256
from weld_pipeline.predict_structured_candidate_model import MODEL_NAME


def main():
    parent=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v3')
    attrs=Path('weld_pipeline/runs/structured_plate_attributes_seed134')
    if not (attrs/'complete.json').exists():
        raise ValueError('Structured training unfinished')
    out=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4');out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():
        raise ValueError('Frozen v4 candidate')
    state=torch.load(parent/'best.pt',map_location='cpu')
    state.update(model_name=MODEL_NAME,attribute_heads=joblib.load(attrs/'heads.joblib'),
        output_schema_version='structured_candidate_attributes_v4',full_contract_pass=False)
    state['source_hashes'].update(state['attribute_heads']['config']['sources'])
    state['source_hashes'].update({p:sha256(p) for p in ['weld_pipeline/predict_structured_candidate_model.py',
        'weld_pipeline/package_structured_candidate_model.py']})
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for path,wanted in {**state['horizontal']['source_hashes'],**state['source_hashes']}.items():
        source=Path(path)
        if not source.exists() or sha256(source)!=wanted:
            source=parent/'source_snapshot'/source.name
        if sha256(source)!=wanted:
            raise ValueError('Required source missing: '+path)
        shutil.copy2(source,archive/source.name)
    torch.save(state,out/'best.pt')
    receipt=json.loads((parent/'training_receipt.json').read_text('utf8'))
    receipt.update(checkpoint_sha256=sha256(out/'best.pt'),parent_checkpoint_sha256=sha256(parent/'best.pt'),
        attribute_candidate_fit=json.loads((attrs/'complete.json').read_text('utf8')),
        output_schema_version='structured_candidate_attributes_v4',raw_verified=False,full_contract_pass=False)
    (out/'training_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print('STRUCTURED_MODEL_PACKAGED',receipt['checkpoint_sha256'],flush=True)


if __name__=='__main__':
    main()
