"""Freeze a new family split after complete cross-role exact-support checks.

Confirmed shared input support joins families conservatively. Historical TRAIN
exposure keeps joined families in TRAIN; protected diagnostics stay outside
fitting and selection. Existing label-version keys and all inventory entries
survive. The old split, models, caches and source data are never overwritten.
"""
import argparse
from collections import Counter
import copy
import hashlib
from pathlib import Path

from tools.qualify_full_dataset_labels import ROOT, BASE
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


def rebuild(original, input_metadata, confirmed_edges):
    families = {row['family_id']: row for row in original['families']}
    parent = {key: key for key in families}

    def find(key):
        if key not in parent:
            raise ValueError('Input provenance references an unknown family')
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(first, second):
        a, b = find(first), find(second)
        parent[max(a, b)] = min(a, b)

    by_input = {row['workpiece_sha256']: row for row in input_metadata}
    for row in input_metadata:
        ids = sorted({version['family_id'] for version in row['versions']})
        for key in ids[1:]:
            join(ids[0], key)
    for first, second in confirmed_edges:
        for a in by_input[first]['versions']:
            for b in by_input[second]['versions']:
                join(a['family_id'], b['family_id'])
    groups = {}
    for key in families:
        groups.setdefault(find(key), []).append(key)
    mapping, revised_families = {}, []
    for members in sorted(groups.values(), key=lambda values: sorted(values)):
        members = sorted(members)
        rows = [families[key] for key in members]
        protected = any(row['previous_role'] == 'PROTECTED_ORIGINAL_DEV' for row in rows)
        if protected:
            split, previous = 'validation', 'PROTECTED_ORIGINAL_DEV'
        elif any(row['split'] == 'train' for row in rows):
            split, previous = 'train', 'GROUPED_HISTORICAL_TRAIN_EXPOSURE'
        elif any(row['split'] == 'validation' for row in rows):
            split, previous = 'validation', 'GROUPED_SELECTION_HOLDOUT'
        else:
            split, previous = 'test', 'GROUPED_FINAL_HOLDOUT'
        fid = members[0] if len(members) == 1 else hashlib.sha256(
            ('FULL_INPUT_SUPPORT_FAMILY_V2:' + ':'.join(members)).encode()).hexdigest()
        key = rows[0]['key'] if len(members) == 1 else 'group_' + fid[:12]
        row = copy.deepcopy(rows[0])
        row.update(family_id=fid, key=key, split=split, previous_role=previous,
            original_family_ids=members, original_previous_roles=[r['previous_role'] for r in rows],
            directories=sorted({d for r in rows for d in r['directories']}),
            pcd_sha256=sorted({sha for r in rows for sha in r['pcd_sha256']}),
            metadata=dict(curved=any(r['metadata']['curved'] for r in rows),
                          children=max(r['metadata']['children'] for r in rows)),
            independent_holdout=False,
            assignment_reason='Confirmed shared input support stays grouped; historical exposure and protection determine use',
            physical_identity_and_complete_history_independence_not_certified=True)
        revised_families.append(row)
        for old in members:
            mapping[old] = row
    examples = []
    for example in original['examples']:
        family = mapping[example['family_id']]
        updated = copy.deepcopy(example)
        updated.update(original_family_id=example['family_id'], original_family_key=example['family_key'],
            original_split=example['split'], family_id=family['family_id'], family_key=family['key'],
            split=family['split'], independent_holdout=False,
            canonical_version=example['directory'] == min(family['directories']))
        examples.append(updated)
    if len({example['key'] for example in examples}) != len(original['examples']):
        raise ValueError('Label versions were merged or omitted')
    for first, second in confirmed_edges:
        used = {mapping[v['family_id']]['split'] for cloud in [first, second] for v in by_input[cloud]['versions']}
        if len(used) != 1:
            raise ValueError('Confirmed input support still crosses partitions')
    if not any(row['split'] == 'test' for row in revised_families):
        raise ValueError('No final holdout remains; do not invent unseen training families')
    if not any(row['split'] == 'validation' and row['previous_role'] != 'PROTECTED_ORIGINAL_DEV' for row in revised_families):
        raise ValueError('No model-selection holdout remains')
    value = copy.deepcopy(original)
    value.update(schema='FULL_FAMILY_DATASET_SPLIT_V2_EXACT_POINT_SUPPORT_REPAIR', families=revised_families,
        examples=examples, family_counts=dict(Counter(row['split'] for row in revised_families)),
        directory_counts=dict(Counter(row['split'] for row in examples)),
        untouched_holdout_family_counts={}, confirmed_input_support_edges_grouped=True,
        old_label_version_keys_and_inventory_entries_preserved=True,
        no_family_crosses_split=True, historical_64_training_forbidden=True,
        input_identity_or_complete_generalization_certified=False,
        confirmed_shared_support_does_not_prove_physical_identity=True,
        arbitrary_rotation_resampling_and_external_history_risks_remain=True)
    return value


def main(args):
    if Path(args.output_name).name != args.output_name or args.output_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('A new independent dataset directory is required')
    if Path(args.exact_audit).name != args.exact_audit or args.exact_audit in ('', '.', '..'):
        raise ValueError('A local complete audit name is required')
    audit = BASE / args.exact_audit
    receipt, protocol = read(audit / 'receipt.json'), read(audit / 'protocol.json')
    if (receipt['status'] != 'ALL_DIRECTED_CROSS_ROLE_EXACT_SUPPORT_CHECKS_COMPLETE'
            or receipt['directed_pairs_checked'] != protocol['every_directed_cross_role_pair_required']
            or receipt['protocol_sha256'] != digest(audit / 'protocol.json')
            or receipt['sources'] != {name: digest(ROOT / name) for name in receipt['sources']}
            or protocol['role_bindings'] != {name: digest(BASE / name) for name in protocol['role_bindings']}):
        raise ValueError('An authentic complete unchanged cross-role audit is required')
    metadata, checked, confirmed = [], set(), []
    records = []
    for item in receipt['results']:
        path = audit / item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Exact support results changed')
        row = read(path); metadata.append(row['target']); records.append(row)
    by_input = {row['workpiece_sha256']: row for row in metadata}
    if len(by_input) != receipt['all_inputs']:
        raise ValueError('Complete unique input provenance required')
    for row in records:
        target = row['target']['workpiece_sha256']
        for source in row['directed_sources']:
            first = source['source_sha256']; pair = first, target
            if pair in checked or first not in by_input or by_input[first]['role'] == row['target']['role']:
                raise ValueError('Invalid or duplicate cross-role pair')
            checked.add(pair)
            if source['full_check_required']:
                full = source['full_point_check']
                if full['source_points_checked'] != by_input[first]['points']:
                    raise ValueError('A containment check omitted original points')
                if full['all_original_points_have_exact_coordinate_support_in_target']:
                    if full['maximum_nearest_distance_mm'] != 0. or full['exact_match_fraction'] != 1.:
                        raise ValueError('Nonzero-distance proximity is not exact support')
                    confirmed.append(pair)
    expected = {(a, b) for a in by_input for b in by_input if by_input[a]['role'] != by_input[b]['role']}
    if checked != expected or len(confirmed) != receipt['confirmed_directed_containments']:
        raise ValueError('The complete directed pair scope was not preserved')
    original = read(BASE / 'split_manifest.json')
    qualification = read(BASE / 'qualification_summary.json')
    known = {v['key'] for row in metadata for v in row['versions']}
    if known != {row['key'] for row in qualification['qualified']}:
        raise ValueError('Input audit omitted a qualified label version')
    value = rebuild(original, metadata, confirmed)
    value.update(source_sha256=digest(__file__), source_bindings={
        str((BASE/'split_manifest.json').relative_to(ROOT)): digest(BASE/'split_manifest.json'),
        str((audit/'receipt.json').relative_to(ROOT)): digest(audit/'receipt.json')})
    out = ROOT / 'server_rebuild' / args.output_name
    out.mkdir(exist_ok=False)
    write(out / 'split_manifest.json', value)
    write(out / 'split_repair_evidence.json', dict(status='CONFIRMED_EXACT_SUPPORT_FAMILIES_REPARTITIONED',
        old_split_sha256=digest(BASE/'split_manifest.json'), new_split_sha256=digest(out/'split_manifest.json'),
        audit_receipt_sha256=digest(audit/'receipt.json'), confirmed_edges=confirmed,
        original_versions=len(original['examples']), revised_versions=len(value['examples']),
        original_families=len(original['families']), revised_families=len(value['families']),
        family_counts=value['family_counts'], directory_counts=value['directory_counts'],
        old_split_or_models_or_data_changed=False, model_fit_calls=0, reference_scoring_calls=0,
        qualified_registries_and_model_training_still_required=True))
    for split in ['train', 'validation', 'test']:
        write(out/(split+'.json'), dict(split=split, split_manifest_sha256=digest(out/'split_manifest.json'),
            families=[r for r in value['families'] if r['split']==split],
            examples=[r for r in value['examples'] if r['split']==split]))
    print('CORRECTED_FAMILY_SPLIT_FROZEN', value['directory_counts'], value['family_counts'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--exact-audit', required=True)
    main(parser.parse_args())
