"""Expand the fixed finite-material audit to every current CAD-backed FIT source."""
import json
import shutil
from pathlib import Path

from training.labels import sha256
from weld_pipeline import audit_candidate_material_identity as audit


def main():
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    reference_root = Path('weld_pipeline/reports/step_contact_continuity_candidates_v1')
    names = tuple(name for name in split['train'] if '__' not in name and
                  (reference_root / (name + '_attribute_candidates.json')).exists())
    if len(names) != 19 or set(names) & set(split['validation']):
        raise ValueError('Expected exactly 19 original CAD-backed FIT sources, no observed checks')
    audit.OUT = Path('weld_pipeline/reports/candidate_material_identity_full_148v2_20261001')
    audit.NAMES = names
    if audit.OUT.exists():
        raise ValueError('Preserve existing full material audit')
    invocation = dict(wrapper_sha256=sha256(__file__), audit_sha256=sha256(audit.__file__),
                      split_sha256=sha256(split_path), names=list(names),
                      parameters_unchanged_from_three_source_audit=True,
                      source_selection_is_not_a_runtime_feature=True, observed_check_read=False)
    invocation_path = audit.OUT.with_name(audit.OUT.name + '_invocation.json')
    if invocation_path.exists():
        raise ValueError('Previous invocation exists; investigate before rerunning')
    invocation_path.write_text(json.dumps(invocation, indent=2, ensure_ascii=False), encoding='utf8')
    audit.main()
    shutil.copy2(__file__, audit.OUT / 'source_snapshot' / Path(__file__).name)
    receipt = dict(invocation=invocation, complete_sha256=sha256(audit.OUT / 'complete.json'),
                   invocation_sha256=sha256(invocation_path), full_contract_pass=False)
    (audit.OUT / 'wrapper_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
