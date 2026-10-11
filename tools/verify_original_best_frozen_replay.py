"""Separate frozen best scores, original-input replay and raw reconstruction.

Every original TRAIN case is required. Original and current parsed reference
views remain separate; no training labels, weights or archived results change.
Original inference runs unchanged into a new directory with audited file reads.
"""
import argparse
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline import sjg_child_parent_model_1256 as inference
from weld_pipeline.sjg_recall_set_worker_1162 import target_for
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, digest, read, write

SOURCES = ['tools/verify_original_best_frozen_replay.py',
    'weld_pipeline/sjg_child_parent_model_1256.py',
    'weld_pipeline/sjg_child_parent_supervision_1255.py',
    'weld_pipeline/sjg_additive_parent_precision_1254.py',
    'weld_pipeline/sjg_parent_set_features_1160.py',
    'weld_pipeline/legacy_incremental_event_runtime_393.py',
    'weld_pipeline/contract_curve_geometry.py',
    'weld_pipeline/sjg_recall_set_worker_1162.py',
    'weld_pipeline/sjg_native_d3_adapter_1056.py',
    'weld_pipeline/contract_geometry_d3_409.py',
    'weld_pipeline/world_set_evaluation_285.py',
    'weld_pipeline/world_set_runtime_285.py',
    'weld_pipeline/server_legacy_runtime.py',
    'weld_pipeline/sjg_local_runtime_904.py']


def plain(value):
    if not value or value in ('.', '..') or Path(value).name != value:
        raise ValueError('A plain independent experiment name is required')
    return value


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@contextlib.contextmanager
def audit_inference_io(destination, forbidden):
    """Capture project input bytes; deny reference/STEP reads and outside writes."""
    original_builtin, original_io = builtins.open, io.open
    reads = {}

    def check(file, mode):
        if isinstance(file, int):
            return
        path = Path(file).resolve()
        if any(flag in mode for flag in ('w', 'a', 'x', '+')):
            path.relative_to(destination.resolve())
            return
        if not path.is_relative_to(ROOT):
            return
        if (path in forbidden or path.suffix.lower() in ('.step', '.stp')
                or 'compatibility' in path.name or 'targets' in path.name):
            raise ValueError('Original inference attempted a reference or STEP read: ' + str(path))
        if path not in reads and path.is_file():
            h = hashlib.sha256()
            with original_builtin(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    h.update(block)
            reads[path] = h.hexdigest()

    def builtin(file, mode='r', *args, **kwargs):
        check(file, mode)
        return original_builtin(file, mode, *args, **kwargs)

    def iofile(file, mode='r', *args, **kwargs):
        check(file, mode)
        return original_io(file, mode, *args, **kwargs)

    builtins.open, io.open = builtin, iofile
    try:
        yield reads
    finally:
        builtins.open, io.open = original_builtin, original_io


def accumulate(total, score):
    for category, values in score['categories'].items():
        row = total.setdefault(category, dict(G=0, predicted=0, TP=0, FP=0))
        for field in row:
            row[field] += values[field]


def reference_core(target):
    return {child['original_child_id']:dict(parent_id=parent.get('original_parent_id'),
        child_count=parent['original_child_count'], is_closed=parent['is_closed'],
        gaps=parent['gaps'], kind=child['original_kind'], weld_type=child['weld_type'],
        representation=child['representation'])
        for parent in target['parents'] for child in parent['children']}


def main(args):
    source = Path(args.model_directory).resolve(strict=True)
    source.relative_to(ROOT)
    receipt = read(source / 'training_receipt.json')
    if digest(source / 'models.pkl') != receipt['model_sha256']:
        raise ValueError('Original registered best model bytes changed')
    keys = [plain(key) for key in receipt['TRAIN']]
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError('All original TRAIN sources must be unique and retained')
    cohort_path = OUT / 'run98/local_cohort.json'
    cohort = read(cohort_path)
    train = read(ROOT / 'server_rebuild' / args.dataset_name / 'training_registry.json')
    forbidden = {(OUT / row['target_file']).resolve() for row in cohort['cases']}
    out = ROOT / 'server_rebuild' / plain(args.experiment_name)
    out.mkdir(exist_ok=False)
    dest = out / 'fresh_inference'
    dest.mkdir()
    for item in ('models.pkl', 'training_receipt.json'):
        (dest / item).symlink_to(source / item)
    sources = {name:digest(ROOT / name) for name in SOURCES}
    protocol = dict(status='ALL_ORIGINAL_TRAIN_FROZEN_OUTPUT_AND_INPUT_REPLAY_WITH_DISTINCT_REFERENCES',
        original_model_directory=str(source.relative_to(ROOT)), original_model_sha256=receipt['model_sha256'],
        original_training_receipt_sha256=digest(source / 'training_receipt.json'),
        original_cohort_sha256=digest(cohort_path), original_cases=keys, sources=sources,
        training_registry_sha256=digest(ROOT / 'server_rebuild' / args.dataset_name / 'training_registry.json'),
        optimizer_updates=0, archived_results_or_training_labels_modified=False,
        raw_xyz_reinference_executed=False, independent_generalization_certified=False)
    write(out / 'protocol.json', protocol)
    totals = {name:{} for name in ('uploaded_on_original', 'uploaded_on_current',
        'fresh_frozen_inputs_on_original', 'fresh_frozen_inputs_on_current',
        'raw_reconstruction_on_original', 'raw_reconstruction_on_current')}
    records = []
    with threadpool_limits(limits=2):
        for key in keys:
            original_target, binding = target_for(key)
            current_folder = ROOT / 'server_rebuild/labels_v1' / key
            current_target = read(current_folder / 'targets.json')
            current_input_folder = ROOT / 'server_rebuild/inputs_v1' / key
            current_input = read(current_input_folder / 'receipt.json')
            matches = [job for job in train['cases'] if job['workpiece_sha256'] == current_input['source_sha256']]
            if len(matches) != 1:
                raise ValueError('Every original baseline input must correspond to one complete TRAIN input')
            original_xyz_path = OUT / binding['input_file']
            current_xyz_path = current_input_folder / 'xyz_mm.npy'
            if (digest(original_xyz_path) != binding['input_sha256']
                    or digest(current_xyz_path) != current_input['xyz_sha256']):
                raise ValueError('Original or current full raw XYZ bytes changed')
            a, b = [np.load(path, mmap_mode='r') for path in (original_xyz_path, current_xyz_path)]
            exact = a.shape == b.shape and all(np.array_equal(a[i:i+100000], b[i:i+100000])
                                               for i in range(0, len(a), 100000))
            if not exact:
                raise ValueError('Original and current full native XYZ inputs differ')
            input_points = len(a)
            del a, b
            frozen = source / ('test_' + key)
            freeze = read(frozen / 'prediction_receipt.json')
            if (freeze['GT_reads'] != 0 or freeze['STEP_reads'] != 0
                    or freeze['model_sha256'] != protocol['original_model_sha256']
                    or digest(frozen / 'final_decoded.json') != freeze['decoded_sha256']):
                raise ValueError('Original uploaded frozen prediction binding changed')
            original_decoded = read(frozen / 'final_decoded.json')
            started = time.perf_counter()
            with audit_inference_io(dest, forbidden) as inputs:
                inference.infer(dest, key)
            for path, sha in inputs.items():
                if digest(path) != sha:
                    raise ValueError('A consumed original inference input changed')
            fresh_folder = dest / ('test_' + key)
            fresh = read(fresh_folder / 'final_decoded.json')
            fresh_receipt = read(fresh_folder / 'prediction_receipt.json')
            if digest(fresh_folder / 'final_decoded.json') != fresh_receipt['decoded_sha256']:
                raise ValueError('Fresh complete frozen-input output differs from its freeze')
            rebuilt = ROOT / 'server_rebuild/best_baseline_restore/legacy_runtime' / ('source_' + key) / 'run446' / ('test_' + key)
            rebuilt_freeze = read(rebuilt / 'prediction_receipt.json')
            if digest(rebuilt / 'final_decoded.json') != rebuilt_freeze['decoded_sha256']:
                raise ValueError('Previous full raw reconstruction output changed')
            rebuilt_decoded = read(rebuilt / 'final_decoded.json')
            scores = {}
            for prefix, decoded in [('uploaded', original_decoded), ('fresh_frozen_inputs', fresh), ('raw_reconstruction', rebuilt_decoded)]:
                for suffix, target in [('original', original_target), ('current', current_target)]:
                    score = score_native_parents(decoded, target, [False] * len(target['parents']))
                    score_name = prefix + '_on_' + suffix
                    scores[score_name] = score
                    accumulate(totals[score_name], score)
            if scores['uploaded_on_original']['categories'] != read(frozen / 'evaluation.json')['score']['categories']:
                raise ValueError('Original uploaded best scoring categories were not reproduced')
            old_core, current_core = reference_core(original_target), reference_core(current_target)
            if old_core.keys() != current_core.keys():
                raise ValueError('Original and current reference child identities differ')
            differences = [dict(child_id=cid, changed_fields={field:dict(original=old_core[cid][field],
                current=current_core[cid][field]) for field in old_core[cid]
                if old_core[cid][field] != current_core[cid][field]})
                for cid in sorted(old_core) if old_core[cid] != current_core[cid]]
            result = dict(case=key, registered_train_job=matches[0]['job_key'],
                complete_raw_xyz_points=input_points, all_points_same_order_and_xyz_exact=True,
                original_target_file=str((OUT / binding['target_file']).relative_to(ROOT)),
                original_target_sha256=binding['target_sha256'], current_target_sha256=digest(current_folder / 'targets.json'),
                uploaded_prediction_sha256=freeze['decoded_sha256'],
                fresh_prediction_sha256=fresh_receipt['decoded_sha256'],
                fresh_output_complete_canonical_json_identical=canonical(fresh) == canonical(original_decoded),
                original_reference_core_differences=differences, scores=scores,
                actual_inference_read_files={str(path.relative_to(ROOT)):sha for path,sha in inputs.items()},
                inference_reference_or_STEP_reads=0, optimizer_updates=0,
                original_frozen_input_replay_seconds=time.perf_counter() - started,
                raw_xyz_reinference_executed=False, full_contract_pass=False)
            write(out / (key + '_assessment.json'), result)
            records.append(dict(case=key, path=str((out / (key + '_assessment.json')).relative_to(ROOT)),
                sha256=digest(out / (key + '_assessment.json')),
                fresh_output_complete_canonical_json_identical=result['fresh_output_complete_canonical_json_identical']))
            print('ORIGINAL_FROZEN_INPUT_REPLAY_COMPLETE', key,
                scores['fresh_frozen_inputs_on_original']['categories']['overall'],
                'COMPLETE_OUTPUT_IDENTICAL', result['fresh_output_complete_canonical_json_identical'], flush=True)
    if sources != {name:digest(ROOT / name) for name in sources}:
        raise ValueError('Bound original replay source changed')
    if totals['uploaded_on_original']['overall']['G'] != receipt['G']:
        raise ValueError('Original full denominator changed')
    summary = dict(status='ALL_ORIGINAL_FROZEN_OUTPUT_AND_FROZEN_INPUT_REPLAYS_ASSESSED_NOT_RAW_REINFERENCE',
        protocol_sha256=digest(out / 'protocol.json'), required_cases=keys, records=records, totals=totals,
        all_fresh_outputs_complete_canonical_json_identical=all(row['fresh_output_complete_canonical_json_identical'] for row in records),
        optimizer_updates=0, current_training_labels_or_weights_modified=False,
        raw_xyz_reinference_executed=False, independent_generalization_certified=False, full_contract_pass=False)
    write(out / 'receipt.json', summary)
    print('ALL_ORIGINAL_BEST_REPLAY_ASSESSED', {k:v['overall'] for k,v in totals.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-directory', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--experiment-name', required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException:
        import traceback
        out = ROOT / 'server_rebuild' / plain(args.experiment_name)
        if out.is_dir():
            write(out / 'failure.json', dict(traceback=traceback.format_exc(), optimizer_updates=0))
        raise
