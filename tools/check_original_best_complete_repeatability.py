"""Repeat the full original frozen-input inference in independent processes.

All five original TRAIN objects, actual candidate decisions, directed parent
assembly, selected curves and CSV round trips are compared. This control uses
the original cached inputs; it is not a raw-XYZ timing or held evaluation.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline import sjg_child_parent_model_1256 as inference
from weld_pipeline.server_full_weld_csv_export import export_decoded, verify_roundtrip
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, digest, read, write
from tools.verify_original_best_frozen_replay import audit_inference_io, canonical, plain, SOURCES as REPLAY_SOURCES
from tools.record_training_experiment import git_head


MODEL = OUT / 'run446'
MODEL_SHA = '6563d8c287b9283e3046b670d8415a68da3b2ed99e59e79275ccf8a06af9fbeb'
SOURCES = list(dict.fromkeys(['tools/check_original_best_complete_repeatability.py',
    'weld_pipeline/server_full_weld_csv_export.py', 'weld_pipeline/runtime_csv_export_287.py',
    *REPLAY_SOURCES]))


def worker(out, attempt):
    folder = out / f'process_{attempt}'
    folder.mkdir(exist_ok=False)
    receipt = read(MODEL / 'training_receipt.json')
    if receipt['model_sha256'] != MODEL_SHA or digest(MODEL / 'models.pkl') != MODEL_SHA:
        raise ValueError('The registered original best weights changed')
    sources = {name: digest(ROOT / name) for name in SOURCES}
    for name in ('models.pkl', 'training_receipt.json'):
        (folder / name).symlink_to(MODEL / name)
    cohort = read(OUT / 'run98/local_cohort.json')
    forbidden = {(OUT / row['target_file']).resolve() for row in cohort['cases']}
    rows = []
    with threadpool_limits(limits=2):
        for key in receipt['TRAIN']:
            plain(key)
            started = time.perf_counter()
            with audit_inference_io(folder, forbidden) as inputs:
                inference.infer(folder, key)
            if any(digest(path) != sha for path, sha in inputs.items()):
                raise ValueError('A frozen original inference input changed')
            actual = folder / ('test_' + key)
            frozen = read(actual / 'prediction_receipt.json')
            decoded = read(actual / 'final_decoded.json')
            original = read(MODEL / ('test_' + key) / 'final_decoded.json')
            if (frozen['GT_reads'] or frozen['STEP_reads']
                    or frozen['decoded_sha256'] != digest(actual / 'final_decoded.json')
                    or canonical(decoded) != canonical(original)):
                raise ValueError('Original complete output was not reproduced without reference reads')
            if decoded['invalid_predictions']:
                raise ValueError('The CSV control cannot silently remove invalid output children')
            # Only the already-verified IO audit supplies this exporter flag.
            exported = dict(decoded, GT_or_STEP_reads=0)
            schema = export_decoded(actual / 'csv', key, exported)
            roundtrip = verify_roundtrip(actual / 'csv', exported)
            assembly = read(actual / 'assembly.json')
            # Solver runtime is measurement metadata. All other fields,
            # including selected direction states, links and objective, remain.
            assembly.pop('seconds', None)
            files = {str(p.relative_to(actual / 'csv')): digest(p)
                     for p in sorted((actual / 'csv').rglob('*')) if p.is_file()}
            row = dict(case=key, decoded_canonical_sha256=canonical(decoded),
                decisions_canonical_sha256=canonical(read(actual / 'decisions.json')),
                assembly_canonical_sha256=canonical(assembly),
                selected_parents=len(decoded['parents']),
                selected_children=decoded['total_prediction_children'],
                csv_rows=schema['published_rows'], csv_files=files, csv_roundtrip=roundtrip,
                consumed_inputs={(('generated:' + str(path.relative_to(folder)))
                    if path.is_relative_to(folder) else str(path.relative_to(ROOT))): sha
                    for path, sha in inputs.items()},
                seconds=time.perf_counter() - started, reference_or_STEP_reads=0)
            rows.append(row)
            write(folder / 'progress.json', dict(case=key, completed=len(rows), cases=len(receipt['TRAIN'])))
            print('ORIGINAL_COMPLETE_REPEAT', attempt, key, row['selected_children'], flush=True)
    if sources != {name: digest(ROOT / name) for name in SOURCES}:
        raise ValueError('Repeatability sources changed during inference')
    result = dict(status='ALL_ORIGINAL_FROZEN_INPUT_COMPLETE_OUTPUTS_AND_CSV_REPLAYED',
        pid=os.getpid(), original_model_sha256=MODEL_SHA, sources=sources,
        original_training_receipt_sha256=digest(MODEL / 'training_receipt.json'),
        rows=rows, fit_calls=0, reference_or_STEP_reads=0, full_contract_pass=False)
    write(folder / 'receipt.json', result)


def main(args):
    out = ROOT / 'server_rebuild/full_dataset_v2' / plain(args.experiment_name)
    if args.worker is not None:
        worker(out, args.worker)
        return
    out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(source_commit=git_head(ROOT), sources=sources,
        original_model_sha256=MODEL_SHA, independent_processes=2,
        scope='All original five TRAIN frozen-input complete outputs; no raw reconstruction or held assessment',
        comparison_metadata_exclusions=['assembly.seconds', 'worker.seconds', 'worker.pid',
                                        'own_generated_output_directory_prefix'],
        reference_or_STEP_reads=0, fit_calls=0, full_contract_pass=False))
    try:
        receipts = []
        for attempt in range(2):
            command = [sys.executable, '-m', 'tools.check_original_best_complete_repeatability',
                       '--experiment-name', args.experiment_name, '--worker', str(attempt)]
            write(out / 'progress.json', dict(stage='INDEPENDENT_COMPLETE_REPLAY', attempt=attempt,
                                             controller_pid=os.getpid()))
            subprocess.run(command, cwd=ROOT, check=True)
            receipts.append(read(out / f'process_{attempt}/receipt.json'))
        if receipts[0]['pid'] == receipts[1]['pid']:
            raise ValueError('Independent inference processes are required')
        comparisons = []
        if len(receipts[0]['rows']) != len(receipts[1]['rows']):
            raise ValueError('One complete object was lost')
        for first, second in zip(receipts[0]['rows'], receipts[1]['rows']):
            fields = ['case', 'decoded_canonical_sha256', 'decisions_canonical_sha256',
                'assembly_canonical_sha256', 'selected_parents', 'selected_children',
                'csv_rows', 'csv_files', 'csv_roundtrip', 'consumed_inputs']
            changed = [name for name in fields if first[name] != second[name]]
            if changed:
                raise ValueError('Complete output repeated differently: ' + first['case'] + ': ' + ', '.join(changed))
            comparisons.append(dict(case=first['case'], compared_fields=fields,
                                    children=first['selected_children'], identical=True))
        if sources != {name: digest(ROOT / name) for name in SOURCES}:
            raise ValueError('Control source binding changed')
        write(out / 'receipt.json', dict(status='PASS_ORIGINAL_BEST_COMPLETE_OUTPUT_AND_CSV_REPEATABILITY',
            protocol_sha256=digest(out / 'protocol.json'), original_model_sha256=MODEL_SHA,
            processes=[r['pid'] for r in receipts], objects=len(comparisons), comparisons=comparisons,
            child_rows_per_process=sum(r['children'] for r in comparisons),
            worker_receipts=[dict(path=f'process_{i}/receipt.json', sha256=digest(out / f'process_{i}/receipt.json'))
                             for i in range(2)], fit_calls=0, reference_or_STEP_reads=0,
            current_V5_repeatability_certified=False, raw_XYZ_latency_certified=False,
            geometry_or_field_accuracy_certified=False, full_contract_pass=False))
        print('PASS_ALL_ORIGINAL_COMPLETE_OUTPUTS_AND_CSV_REPEATABILITY', flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--worker', type=int, choices=(0, 1))
    main(parser.parse_args())
