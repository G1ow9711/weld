"""Run the original label qualifier in a new authenticated family partition.

The original extraction algorithm executes again against unchanged raw data.
Every existing geometry, attribute and signed-frame artifact is compared with
the original qualification. Only the target's assigned role may change.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from tools import qualify_full_dataset_labels as original
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = original.ROOT
OLD = original.BASE


def main(args):
    if Path(args.dataset_name).name != args.dataset_name or args.dataset_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('An independent repaired dataset directory is required')
    base = ROOT / 'server_rebuild' / args.dataset_name
    started = time.perf_counter()
    repair = read(base / 'split_repair_evidence.json')
    if (repair['status'] != 'CONFIRMED_EXACT_SUPPORT_FAMILIES_REPARTITIONED'
            or repair['old_split_sha256'] != digest(OLD / 'split_manifest.json')
            or repair['new_split_sha256'] != digest(base / 'split_manifest.json')):
        raise ValueError('The original complete input-support split repair is required')
    bindings = {str(p.relative_to(ROOT)): digest(p) for p in [
        OLD/'qualification_summary.json', OLD/'split_manifest.json', base/'split_manifest.json',
        base/'split_repair_evidence.json']}
    sources = {name: digest(ROOT/name) for name in ['tools/qualify_repaired_full_dataset_labels.py',
        'tools/qualify_full_dataset_labels.py']}
    out = base / 'qualification_repair'
    out.mkdir(exist_ok=False)
    write(out/'protocol.json', dict(bindings=bindings, sources=sources,
        original_qualification_algorithm_executes=True, only_dataset_output_and_assignments_reconfigured=True,
        original_data_labels_models_or_input_caches_modified=False, model_fit_calls=0, model_reference_scores=0))
    previous_base = original.BASE
    try:
        original.BASE = base
        original.main()
        old, new = read(OLD/'qualification_summary.json'), read(base/'qualification_summary.json')
        by_key = {row['key']: row for row in old['qualified']}
        if (set(by_key) != {row['key'] for row in new['qualified']}
                or {row['key'] for row in old['quarantined']} != {row['key'] for row in new['quarantined']}):
            raise ValueError('A qualified or quarantined source version changed eligibility')
        compared = []
        for row in new['qualified']:
            key = row['key']; before = by_key[key]
            first, second = OLD/'labels'/key, base/'labels'/key
            if row['source_files'] != before['source_files'] or row['source_sha256'] != sources['tools/qualify_full_dataset_labels.py']:
                raise ValueError('Raw source identity or original qualification implementation changed')
            a, b = read(first/'targets.json'), read(second/'targets.json')
            a.pop('role'); b.pop('role')
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                raise ValueError('Repartition changed ordered reference geometry')
            for name in ['endpoint_evidence.json', 'attribute_records.json']:
                if (first/name).read_bytes() != (second/name).read_bytes():
                    raise ValueError('Attributes or original signed endpoint frames changed')
            with np.load(first/'attribute_labels.npz', allow_pickle=False) as a, np.load(second/'attribute_labels.npz', allow_pickle=False) as b:
                if a.files != b.files:
                    raise ValueError('Attribute arrays were omitted or renamed')
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])
                    if a[name].dtype != b[name].dtype or a[name].tobytes() != b[name].tobytes():
                        raise ValueError('Typed attribute values or signed zero bits changed')
            compared.append(dict(key=key, original_receipt_sha256=digest(first/'receipt.json'),
                repaired_receipt_sha256=digest(second/'receipt.json'), original_split=before['split'],
                repaired_split=row['split'], geometry_attributes_and_raw_frames_unchanged=True))
        if sources != {name:digest(ROOT/name) for name in sources} or bindings != {name:digest(ROOT/name) for name in bindings}:
            raise ValueError('Qualification source or frozen input bindings changed')
        write(out/'receipt.json', dict(status='ALL_REPAIRED_VERSIONS_REQUALIFIED_WITH_GEOMETRY_ATTRIBUTES_AND_FRAMES_PRESERVED',
            sources=sources, protocol_sha256=digest(out/'protocol.json'), compared=compared,
            qualified_versions=len(compared), quarantine_versions=len(new['quarantined']),
            repaired_qualification_sha256=digest(base/'qualification_summary.json'),
            qualified_counts_by_split=new['qualified_directories_by_split'],
            qualified_families_by_split=new['qualified_families_by_split'],
            original_data_labels_or_model_weights_changed=False, model_fit_calls=0,
            independent_generalization_certified=False, seconds=time.perf_counter()-started))
        print('ALL_REPAIRED_QUALIFICATION_VERIFIED', len(compared), new['qualified_directories_by_split'], flush=True)
    except Exception:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), unix_time=time.time()))
        raise
    finally:
        original.BASE = previous_base


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    main(parser.parse_args())
