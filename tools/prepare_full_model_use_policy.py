"""Separate training, selection, final scoring and protected diagnostic uses.

The original split stays unchanged. Historical role evidence is retained but
does not become a certificate of absence of later development exposure.
"""
from collections import Counter
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main():
    outputs = [BASE / name for name in ('model_use_policy.json', 'validation_selection.json', 'diagnostic.json')]
    if any(path.exists() for path in outputs):
        raise ValueError('Use policy already exists; do not overwrite a frozen version')
    manifest = read(BASE / 'split_manifest.json')
    qualification = read(BASE / 'qualification_summary.json')
    qualified = {row['key'] for row in qualification['qualified']}
    split_sha = digest(BASE / 'split_manifest.json')
    if qualification['split_manifest_sha256'] != split_sha:
        raise ValueError('Qualification split binding changed')
    families = {row['family_id']: row for row in manifest['families']}
    examples = []
    for example in manifest['examples']:
        family = families[example['family_id']]
        protected = family['previous_role'] == 'PROTECTED_ORIGINAL_DEV'
        accepted = example['key'] in qualified
        split = example['split']
        if protected and split != 'validation':
            raise ValueError('Protected diagnostic assignment changed')
        examples.append(dict(key=example['key'], family_id=example['family_id'], family_key=example['family_key'],
                             directory=example['directory'], assigned_split=split, qualified=accepted,
                             fit_allowed=accepted and split == 'train' and not protected,
                             selection_allowed=accepted and split == 'validation' and not protected,
                             final_evaluation_allowed=accepted and split == 'test' and not protected,
                             diagnostic_only=protected, exposure_evidence_sha256=split_sha,
                             exposure_evidence_scope='Frozen source manifests and previous role; complete later-use audit not certified',
                             independent_generalization_certified=False))
    if any(row['fit_allowed'] and (row['selection_allowed'] or row['final_evaluation_allowed']) for row in examples):
        raise ValueError('Model use roles overlap')
    selected = [row for row in examples if row['selection_allowed']]
    diagnostic = [row for row in examples if row['diagnostic_only']]
    if any(row['family_key'] == 'new_f7b2ad7ae982' for row in selected):
        raise ValueError('Protected original 64 entered model selection')
    policy = dict(status='EXPLICIT_MODEL_USE_POLICY_NOT_INDEPENDENCE_CERTIFICATE', examples=examples,
                  split_manifest_sha256=split_sha, qualification_summary_sha256=digest(BASE / 'qualification_summary.json'),
                  split_roles_unchanged=True, fit_calls=0, model_predictions_scored=0,
                  independent_generalization_certified=False,
                  later_exposure_and_nonbyte_input_identity_audits_still_required=True,
                  same_input_geometry_variants_require_reference_policy_before_combining_set_losses=True,
                  consultation_or_quarantine_does_not_authorize_test_for_selection=True,
                  model_use_counts={field: sum(row[field] for row in examples)
                                    for field in ('fit_allowed', 'selection_allowed', 'final_evaluation_allowed', 'diagnostic_only')},
                  source_sha256=digest(__file__))
    write(outputs[0], policy)
    binding = dict(split_manifest_sha256=split_sha, model_use_policy_sha256=digest(outputs[0]),
                   fit_allowed=False, independence_certified=False)
    write(outputs[1], dict(role='MODEL_SELECTION_ONLY', examples=selected,
                          families=sorted({row['family_id'] for row in selected}),
                          protected_diagnostic_excluded=True, **binding))
    write(outputs[2], dict(role='PROTECTED_DIAGNOSTIC_ONLY', examples=diagnostic,
                          families=sorted({row['family_id'] for row in diagnostic}),
                          selection_allowed=False, final_evaluation_allowed=False, **binding))
    print('MODEL_USE_POLICY_FROZEN', policy['model_use_counts'], flush=True)


if __name__ == '__main__':
    main()
