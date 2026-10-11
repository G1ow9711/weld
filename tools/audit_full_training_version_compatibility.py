"""Audit same-input training views without merging or rewriting their labels.

Strictly identical ordered geometry permits identity-independent field checks.
Different geometry, direction, or ambiguous duplicate parents require a declared
reference policy, not contradictory background targets on the same XYZ input.
This audit does not certify equivalence of resampled or cropped input clouds.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def normalized(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        # Exact numeric identity, including exports which use 0 versus 0.0.
        number = float(value)
        return ['number', (0.0 if number == 0 else number).hex()]
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in sorted(value.items())}
    raise TypeError('Unsupported geometry value')


def parent_identity(parent):
    geometry = dict(is_closed=parent['is_closed'], gaps=parent['gaps'],
                    children=[dict(kind=child['original_kind'], weld_type=child['weld_type'],
                                   representation=child['representation'])
                              for child in parent['children']])
    # Parent export row order/IDs are irrelevant; child order and orientation
    # are deliberately retained. Whole-parent reversal is not auto-aligned.
    return json.dumps(normalized(geometry), sort_keys=True, separators=(',', ':'))


def view(version):
    folder = ROOT / version['label_directory']
    if digest(folder / 'receipt.json') != version['label_receipt_sha256']:
        raise ValueError('Qualified supervision receipt changed')
    receipt = read(folder / 'receipt.json')
    if receipt['role'] != 'TRAIN' or receipt['family_key'] == 'new_f7b2ad7ae982':
        raise ValueError('Held or diagnostic supervision entered audit')
    for name, expected in receipt['artifacts'].items():
        if digest(folder / name) != expected:
            raise ValueError('Qualified supervision artifact changed')
    target = read(folder / 'targets.json')
    evidence = {parent['parent_id']: parent for parent in read(folder / 'endpoint_evidence.json')['parents']}
    parents = defaultdict(list)
    for parent in target['parents']:
        key = parent_identity(parent)
        exported = evidence[parent['original_parent_id']]
        if [c['original_child_id'] for c in parent['children']] != [c['child_id'] for c in exported['children']]:
            raise ValueError('Ordered attribute identity does not match geometry')
        parents[key].append(exported)
    return parents


def fields(parents):
    result = {}
    for geometry, matching in parents.items():
        if len(matching) != 1:
            continue  # Duplicate geometry is not an attribute identity proof.
        owner = hashlib.sha256(geometry.encode()).hexdigest()
        for index, child in enumerate(matching[0]['children']):
            for side in ('start', 'end'):
                for name, label in child['endpoint_attributes'][side].items():
                    if label['known']:
                        value = (label['encoding'], normalized(label['raw_value']) if label['encoding'] == 'EXACT_MM' else None)
                        result[owner, index, side, name] = value
                frame = child['endpoint_frames'][side]
                if frame['known']:
                    result[owner, index, side, 'export_frame_axes'] = normalized(frame['raw_export_axes'])
    return result


def main():
    destination = BASE / 'training_version_compatibility.json'
    if destination.exists():
        raise ValueError('Audit already exists; preserve it and create a new version for changed evidence')
    registry = read(BASE / 'training_registry.json')
    rows = []
    for job in registry['cases']:
        views = [view(version) for version in job['versions']]
        identities = [Counter({key: len(value) for key, value in parents.items()}) for parents in views]
        same_geometry = all(identity == identities[0] for identity in identities)
        ambiguous = any(count > 1 for identity in identities for count in identity.values())
        differences = []
        if same_geometry and not ambiguous:
            known = defaultdict(list)
            for version, parents in zip(job['versions'], views):
                for field, value in fields(parents).items():
                    known[field].append((version['key'], value))
            for field, values in known.items():
                if any(value != values[0][1] for _, value in values):
                    owner, index, side, name = field
                    differences.append(dict(parent_geometry_sha256=owner, child_index=index, endpoint=side,
                                            field=name, versions=[key for key, _ in values],
                                            reason='KNOWN_EXPORT_VALUES_DIFFER_EXACTLY_REQUIRES_FIELD_POLICY'))
        if len(views) == 1:
            status = 'SINGLE_VERSION'
        elif not same_geometry:
            status = 'GEOMETRY_VARIANTS_REQUIRE_ALIGNMENT_OR_REFERENCE_POLICY'
        elif ambiguous:
            status = 'AMBIGUOUS_IDENTICAL_PARENT_ATTRIBUTE_ALIGNMENT'
        elif differences:
            status = 'SAME_GEOMETRY_KNOWN_FIELD_DIFFERENCES_REQUIRE_POLICY'
        else:
            status = 'STRICT_GEOMETRY_EQUIVALENT_NO_KNOWN_FIELD_CONFLICT'
        rows.append(dict(job_key=job['job_key'], workpiece_sha256=job['workpiece_sha256'],
                         versions=[v['key'] for v in job['versions']], status=status,
                         strict_geometry_equal_ignoring_parent_ids_and_export_row_order=same_geometry,
                         field_differences=differences, versions_retained=True,
                         independent_child_or_whole_parent_reversal_auto_aligned=False,
                         references_can_create_shared_physical_negative_labels=False))
    value = dict(status='ALL_TRAINING_VIEWS_AUDITED_WITHOUT_RELABELING',
                 registry_sha256=digest(BASE / 'training_registry.json'), rows=rows,
                 clouds=len(rows), versions=sum(len(row['versions']) for row in rows),
                 status_counts=dict(Counter(row['status'] for row in rows)),
                 field_difference_count=sum(len(row['field_differences']) for row in rows),
                 geometry_not_exact_does_not_prove_physical_conflict=True,
                 numeric_differences_not_silently_rounded_or_averaged=True,
                 known_optional_attributes_also_audited=True,
                 labels_modified=False, label_versions_removed=0, fit_calls=0,
                 validation_or_test_label_reads=0, source_sha256=digest(__file__))
    write(destination, value)
    print('ALL_TRAINING_VERSION_COMPATIBILITY_AUDITED', value['status_counts'],
          'known field differences', value['field_difference_count'], flush=True)


if __name__ == '__main__':
    main()
