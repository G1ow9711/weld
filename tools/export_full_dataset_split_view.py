"""Expose the existing complete family split without moving or relabeling data."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'data/G_Data'
INDEX_FILES = ('split_manifest.json', 'qualification_summary.json',
               'training_registry.json', 'evaluation_registry.json',
               'model_use_policy.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local_name(value):
    if not value or value in ('.', '..') or Path(value).name != value:
        raise ValueError('A plain local directory name is required')
    return value


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')


def table(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def contained(root, value):
    path = (root / value).resolve(strict=True)
    path.relative_to(root.resolve())
    return path


def main(args):
    base = contained(ROOT / 'server_rebuild', local_name(args.dataset_name))
    bindings = {name: digest(base / name) for name in INDEX_FILES}
    split = read(base / 'split_manifest.json')
    qualification = read(base / 'qualification_summary.json')
    training = read(base / 'training_registry.json')
    held = read(base / 'evaluation_registry.json')
    policy = {row['key']: row for row in read(base / 'model_use_policy.json')['examples']}
    proof = read(base / 'registration_repair/receipt.json')
    if any(digest(base / name) != sha for name, sha in proof['bindings'].items()):
        raise ValueError('Registered data split changed')
    qualified = {row['key']: row for row in qualification['qualified']}
    quarantined = qualification['quarantined']
    assigned = {row['key']: row for row in split['examples']}
    if (len(qualified) != len(qualification['qualified'])
            or set(qualified) & {row['key'] for row in quarantined}
            or set(assigned) != set(qualified) | {row['key'] for row in quarantined}):
        raise ValueError('Source inventory is omitted or duplicated')

    progress_path = base / args.parent_training_name / 'training_progress.json'
    progress = read(progress_path) if progress_path.exists() else {}
    visits = Counter()
    for path in (base / args.parent_training_name / 'job_records').glob('*/*/receipt.json'):
        receipt = read(path)
        if receipt['status'] != 'COMPLETE_REGISTERED_JOB_FIT_RECORDED_NOT_MODEL_PROMOTION':
            raise ValueError('Incomplete selector job was counted as trained')
        visits[receipt['job_key']] += 1
    query_path = base / args.query_training_name / 'receipt.json'
    query = read(query_path) if query_path.exists() else {}
    query_epochs = query.get('epochs', 0)
    if query_epochs and (query.get('training_versions') != len(qualified.keys() & {
            version['key'] for case in training['cases'] for version in case['versions']})
            or query.get('status') != 'ALL_TRAIN_REFERENCE_RECALL_FITTED_NOT_PROMOTED_NOT_COMPLETE_INFERENCE'
            or query.get('completed_views') != query_epochs * training['qualified_training_versions']):
        raise ValueError('Complete query training receipt does not cover every TRAIN version')

    rows, families, cloud_roles, seen = [], defaultdict(set), defaultdict(set), set()
    for registry, train_role in ((training, True), (held, False)):
        for case in registry['cases']:
            cloud = contained(DATA, case['workpiece_source'])
            shard = contained(base / ('input_shards' if train_role else 'evaluation_input_shards'),
                              case['job_key'])
            if not (shard / 'receipt.json').is_file():
                raise ValueError('A complete registered input is not ready')
            for version in case['versions']:
                key = version['key']
                if key in seen or key not in qualified:
                    raise ValueError('Qualified version is duplicated or absent')
                seen.add(key)
                source = qualified[key]
                assignment = assigned[key]
                if version['split'] != assignment['split'] or source['split'] != assignment['split']:
                    raise ValueError('A view crossed its assigned split')
                role = 'TRAIN' if train_role else version['role']
                if policy[key]['fit_allowed'] != train_role:
                    raise ValueError('The manifest would permit held data to train')
                labels = contained(ROOT, version['label_directory'])
                label_receipt = labels / 'receipt.json'
                if digest(label_receipt) != version['label_receipt_sha256']:
                    raise ValueError('Qualified label receipt changed')
                raw_sha = source['source_files'][source['workpiece_source']]
                if raw_sha != case['workpiece_sha256']:
                    raise ValueError('Registered raw XYZ ownership differs from qualified data')
                families[version['family_id']].add(version['split'])
                cloud_roles[raw_sha].add(version['split'])
                rows.append(dict(split=version['split'], key=key, family_id=version['family_id'],
                    role=role, fit_allowed=train_role,
                    selection_allowed=policy[key]['selection_allowed'],
                    raw_xyz_path=str(cloud), raw_xyz_sha256=raw_sha,
                    labels_path=str(labels), prepared_input_path=str(shard),
                    source_directory=str(contained(DATA, version['directory'])),
                    parents=source['parents'], children=source['children'],
                    query_training_epochs=query_epochs if train_role else 0,
                    selector_completed_visits=visits[case['job_key']] if train_role else 0,
                    required_attribute_labels_complete=source['required_attributes_complete']))
    if seen != set(qualified) or any(len(value) != 1 for value in families.values()) \
            or any(len(value) != 1 for value in cloud_roles.values()):
        raise ValueError('Complete version coverage or family/input separation failed')
    if bindings != {name: digest(base / name) for name in INDEX_FILES}:
        raise ValueError('Dataset changed while the browse view was built')

    out = base / local_name(args.output_name)
    out.mkdir(exist_ok=False)
    fields = list(rows[0])
    for subset in ('train', 'validation', 'test'):
        folder = out / subset
        folder.mkdir()
        members = sorted((row for row in rows if row['split'] == subset), key=lambda row: row['key'])
        table(out / (subset + '.csv'), members, fields)
        (out / (subset + '.txt')).write_text('\n'.join(row['key'] for row in members) + '\n',
                                           encoding='utf-8')
        for row in members:
            entry = folder / row['key']
            entry.mkdir()
            for name, field in (('raw_xyz.ply', 'raw_xyz_path'), ('labels', 'labels_path'),
                                ('prepared_input', 'prepared_input_path'), ('source', 'source_directory')):
                target = Path(row[field])
                (entry / name).symlink_to(target, target_is_directory=target.is_dir())
            write(entry / 'assignment.json', row)
    table(out / 'all_qualified.csv', sorted(rows, key=lambda row: (row['split'], row['key'])), fields)
    table(out / 'quarantine.csv', quarantined, ('key', 'family_id', 'directory', 'split', 'reason'))
    write(out / 'quarantine.json', quarantined)
    write(out / 'utilization_snapshot.json', dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        query_receipt_path=str(query_path), query_receipt_sha256=digest(query_path) if query else None,
        query_complete_epochs=query_epochs, selector_progress=progress,
        selector_visit_counts=dict(visits),
        evaluation_inputs_ready=True, evaluated_model_accuracy_not_claimed=True,
        note='This snapshot records actual completed training; validation and test are used for evaluation, never fitting.'))
    for subset in ('train', 'validation', 'test'):
        expected = {row['key'] for row in rows if row['split'] == subset}
        if {path.name for path in (out / subset).iterdir()} != expected:
            raise ValueError('Browsable split lost a qualified data version')
        with (out / (subset + '.csv')).open(encoding='utf-8-sig', newline='') as stream:
            restored = list(csv.DictReader(stream))
        if {row['key'] for row in restored} != expected or len(restored) != len(expected):
            raise ValueError('CSV readback lost or duplicated a qualified data version')
    broken = [str(path) for path in out.rglob('*') if path.is_symlink() and not path.exists()]
    if broken:
        raise ValueError('Broken split links: ' + repr(broken))
    summary = dict(status='ALL_QUALIFIED_DATA_BROWSEABLE_IN_ORIGINAL_SPLITS',
        recorded_utc=datetime.now(timezone.utc).isoformat(), source_bindings=bindings,
        exporter_sha256=digest(Path(__file__)), qualified_versions=len(rows),
        quarantined_versions=len(quarantined),
        version_counts=dict(Counter(row['split'] for row in rows)),
        children_by_split={subset: sum(row['children'] for row in rows if row['split'] == subset)
                           for subset in ('train', 'validation', 'test')},
        family_counts={subset: sum(value == {subset} for value in families.values())
                       for subset in ('train', 'validation', 'test')},
        unique_input_clouds=len(cloud_roles), all_inventory_versions_accounted=True,
        all_registered_input_shards_ready=True, family_and_raw_cloud_split_overlap=False,
        copied_raw_data_bytes=0, original_data_roles_or_models_modified=False,
        held_reference_geometry_parsed=False, model_predictions_or_fit_calls=0,
        full_contract_pass=False)
    write(out / 'receipt.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--output-name', default='dataset_splits')
    parser.add_argument('--query-training-name', default='full_weld_reference_recall_v3', type=local_name)
    parser.add_argument('--parent-training-name', default='full_parent_dataset_recall_final_v1', type=local_name)
    main(parser.parse_args())
