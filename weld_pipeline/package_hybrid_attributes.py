"""Retain geometry weights exactly while adding a candidate attribute head."""
import json
import shutil
from pathlib import Path

import joblib
import torch

from training.labels import sha256
from weld_pipeline.predict_hybrid_with_attributes import MODEL_NAME


def main():
    geometry_root=Path('weld_pipeline/runs/contract_full_fit_hybrid_v1')
    attribute_root=Path('weld_pipeline/runs/derived_plate_attributes_seed130')
    if not (attribute_root/'complete.json').exists():
        raise ValueError('Attribute training incomplete')
    state=torch.load(geometry_root/'best.pt',map_location='cpu')
    attributes=joblib.load(attribute_root/'heads.joblib')
    out=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v1')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():
        raise ValueError('Frozen attribute checkpoint exists')
    sources={**state['source_hashes'],**attributes['config']['sources'],
             **{p:sha256(p) for p in ['weld_pipeline/predict_hybrid_with_attributes.py','weld_pipeline/package_hybrid_attributes.py']}}
    state.update(model_name=MODEL_NAME,attribute_heads=attributes,source_hashes=sources,
                 geometry_checkpoint_sha256=sha256(geometry_root/'best.pt'),
                 attribute_scope='Unreviewed CAD-derived candidate fit; two piece sources only, no independent E2 acceptance.',
                 full_contract_pass=False)
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for name,wanted in {**state['horizontal']['source_hashes'],**sources}.items():
        source=Path(name)
        if not source.exists() or sha256(source)!=wanted:
            source=geometry_root/'source_snapshot'/source.name
        if sha256(source)!=wanted:
            raise ValueError('Missing trained source snapshot')
        shutil.copy2(source,archive/source.name)
    torch.save(state,out/'best.pt')
    receipt={'status':'DEVELOPMENT_SINGLE_CHECKPOINT_FULL_CONTRACT_NOT_ACHIEVED',
        'checkpoint_sha256':sha256(out/'best.pt'),'geometry_parent_sha256':sha256(geometry_root/'best.pt'),
        'attribute_heads_sha256':sha256(attribute_root/'heads.joblib'),'geometry_weights_changed':False,
        'geometry_metrics':json.loads((geometry_root/'training_receipt.json').read_text('utf8'))['roles'],
        'attribute_candidate_fit':json.loads((attribute_root/'complete.json').read_text('utf8')),
        'runtime_step_forbidden':True,'full_contract_pass':False,'raw_verified':False}
    (out/'training_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print('PACKAGED_ATTRIBUTE_CHECKPOINT',sha256(out/'best.pt'),flush=True)


if __name__=='__main__':
    main()
