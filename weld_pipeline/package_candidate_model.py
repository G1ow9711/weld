"""Final candidate package, preserving geometry and learned attribute weights."""
import json
import shutil
from pathlib import Path

import torch
from training.labels import sha256


def main():
    parent=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v1')
    out=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v2')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():
        raise ValueError('Frozen candidate package exists')
    state=torch.load(parent/'best.pt',map_location='cpu')
    state['source_hashes'].update({p:sha256(p) for p in ['weld_pipeline/predict_candidate_model.py','weld_pipeline/package_candidate_model.py']})
    state['output_schema_version']='candidate_attributes_v2'
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for name,wanted in {**state['horizontal']['source_hashes'],**state['source_hashes']}.items():
        source=Path(name)
        if not source.exists() or sha256(source)!=wanted:
            source=parent/'source_snapshot'/source.name
        if sha256(source)!=wanted:
            raise ValueError('Candidate source archive inconsistent')
        shutil.copy2(source,archive/source.name)
    torch.save(state,out/'best.pt')
    receipt=json.loads((parent/'training_receipt.json').read_text('utf8'))
    receipt.update(checkpoint_sha256=sha256(out/'best.pt'),parent_checkpoint_sha256=sha256(parent/'best.pt'),
        output_schema_version='candidate_attributes_v2',geometry_weights_changed=False,attribute_weights_changed=False,raw_verified=False)
    (out/'training_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print('CANDIDATE_PACKAGED',sha256(out/'best.pt'),flush=True)


if __name__=='__main__':
    main()
