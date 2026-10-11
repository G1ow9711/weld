"""Package fixed 146 height with preserved v4 geometry and plate attributes."""
import json
import shutil
from pathlib import Path

import joblib
import torch

from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.predict_structured_height_candidate import MODEL_NAME


def main():
    root=Path('weld_pipeline/runs/contract_structured_height_candidate_v5')
    if root.exists():
        raise ValueError('Fixed candidate v5 already exists')
    parent=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt')
    heads=Path('weld_pipeline/runs/continuous_local_height_seed146/heads.joblib')
    state=torch.load(parent,map_location='cpu')
    verify_cached_inference_sources(state,parent);verify_cached_inference_sources(state['horizontal'],parent)
    check=json.loads(Path('weld_pipeline/reports/fresh_point_context_20261001.json').read_text('utf8'))
    if not check['passed'] or check['source_sha256']!=sha256('weld_pipeline/fresh_point_context.py'):
        raise ValueError('Current H/V context consistency check required')
    for name in ('224-GR24A1','207-FR79M'):
        verification=Path('weld_pipeline/reports/continuous_raw_v2_20261001')/('raw_'+name)/'raw_verification.json'
        receipt=json.loads(verification.read_text('utf8'))
        if not receipt['sources_unchanged'] or not receipt['input_unchanged'] or receipt['negative_controls']!=3:
            raise ValueError('FIT raw isolation receipt required')
    old_sources=dict(state['source_hashes']);height=joblib.load(heads)
    sources=[__file__,'weld_pipeline/predict_structured_height_candidate.py','weld_pipeline/fresh_point_context.py',
        'weld_pipeline/continuous_contextual_height_model.py','weld_pipeline/continuous_contact_sampling.py',
        'weld_pipeline/contextual_local_height_model.py','weld_pipeline/finite_contact_plate_profiles.py',
        'weld_pipeline/contract_export.py','weld_pipeline/predict_hybrid_with_attributes.py']
    state.update(model_name=MODEL_NAME,height_heads=height,parent_geometry_checkpoint_sha256=sha256(parent),
        height_training_heads_sha256=sha256(heads),height_definition='实际接触板及局部底板基准',
        development_candidate_only=True,full_contract_pass=False,customer_attribute_truth_reviewed=False)
    state['source_hashes'].update({source:sha256(source) for source in sources})
    root.mkdir(parents=True);snapshot=root/'source_snapshot';snapshot.mkdir()
    all_sources={**state['horizontal']['source_hashes'],**state['source_hashes'],**height['config']['sources']}
    for source,digest in all_sources.items():
        file=Path(source)
        if not file.exists() or sha256(file)!=digest:
            file=parent.parent/'source_snapshot'/file.name
        if not file.exists() or sha256(file)!=digest:
            raise ValueError('Required source snapshot missing: '+source)
        shutil.copy2(file,snapshot/file.name)
    checkpoint=root/'candidate.pt';torch.save(state,checkpoint)
    verify_cached_inference_sources(state,checkpoint)
    result=dict(model_name=MODEL_NAME,checkpoint_sha256=sha256(checkpoint),
        parent_geometry_checkpoint_sha256=sha256(parent),height_heads_sha256=sha256(heads),
        source_sha256=state['source_hashes'],old_source_hashes_preserved=all(state['source_hashes'][key]==value for key,value in old_sources.items()),
        geometry_or_thresholds_retrained=False,original_pcd_excel_step_or_labels_modified=False,
        default_v4_entrypoint_replaced=False,full_contract_pass=False,raw_export_check_status='PENDING')
    (root/'combination_receipt.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('STRUCTURED_HEIGHT_CANDIDATE_SAVED',str(checkpoint),sha256(checkpoint),flush=True)


if __name__=='__main__':
    main()
