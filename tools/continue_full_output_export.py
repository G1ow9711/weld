"""Export all held roles after the existing validation-fixed TEST completes.

This read-only continuation never fits, selects another model or reads label
geometry. It authenticates the same final model pair and all registered output
records, and calls the independently verified canonical CSV exporter.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.export_full_weld_predictions import SOURCES as EXPORT_SOURCES
from tools.continue_full_recall_training import held_stages
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.record_training_experiment import git_head

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCES = list(dict.fromkeys([*EXPORT_SOURCES, 'tools/continue_full_output_export.py',
    'tools/verify_full_weld_csv_export.py', 'tools/continue_validation_fixed_test.py',
    'tools/continue_full_recall_training.py']))


def name(value):
    if not value or value in ('.', '..') or Path(value).name != value:
        raise ValueError('Independent local experiment names required')
    return value


def atomic(path, value):
    pending = path.with_name(path.name + '.incomplete')
    write(pending, value); os.replace(pending, path)


def admission(semantic_path, actual_path):
    for path in [semantic_path, actual_path]:
        path.relative_to(ROOT)
    semantic, actual = read(semantic_path), read(actual_path)
    if (semantic['status'] != 'PASS_COMPLETE_MODEL_CSV_SEMANTICS_AND_SIGNED_ROUNDTRIP'
            or not semantic['signed_zero_and_axis_roundtrip_checked']
            or not semantic['changed_signed_axis_rejected'] or not semantic['parent_Any_not_broadcast']
            or not semantic['unknown_blank_not_zero'] or semantic['reference_geometry_reads'] != 0
            or semantic['fit_calls'] != 0
            or semantic['sources'] != {p: digest(ROOT / p) for p in semantic['sources']}):
        raise ValueError('Current complete CSV semantic controls required')
    if (actual['status'] != 'ALL_AUTHENTICATED_FROZEN_OUTPUTS_CSV_EXPORTED_NOT_CERTIFIED'
            or actual['role'] != 'TRAIN' or actual['exported_records'] < 1
            or actual['exported_child_rows'] < 1 or actual['reference_geometry_reads'] != 0
            or actual['fit_calls'] != 0
            or not actual['every_original_parent_and_child_and_signed_frame_preserved']):
        raise ValueError('Actual nonempty authenticated TRAIN export control required')
    protocol_path = actual_path.parent / 'protocol.json'
    protocol = read(protocol_path)
    if (digest(protocol_path) != actual['protocol_sha256']
            or protocol['sources'] != {p: digest(ROOT / p) for p in protocol['sources']}):
        raise ValueError('Actual export source and protocol binding changed')
    for item in actual['records']:
        path = (ROOT / item['path']).resolve(); path.relative_to(actual_path.parent)
        if digest(path) != item['sha256'] or not read(path)['roundtrip']['ordered_positions_and_signed_frames_bitwise_equal']:
            raise ValueError('Actual complete signed roundtrip evidence changed')
    storage_path = actual_path.parent / 'archive_storage_binding.json'
    if storage_path.exists():
        storage = read(storage_path)
        archive = (ROOT / storage['archive_path']).resolve(); archive.relative_to(ROOT)
        if (storage['status'] != 'COMPLETE_TRAIN_CSV_BUNDLE_ALL_ORIGINAL_BYTES_VERIFIED'
                or not storage['all_members_read_back_and_sha_verified']
                or storage['original_export_receipt_sha256'] != digest(actual_path)
                or storage['original_export_protocol_sha256'] != digest(protocol_path)
                or storage['archive_sha256'] != digest(archive)):
            raise ValueError('Original TRAIN export bundle was not preserved')


def main(args):
    if args.poll_seconds < 5:
        raise ValueError('Conservative explicit polling interval required')
    final = BASE / name(args.final_test_name)
    out = BASE / name(args.experiment_name); out.mkdir(exist_ok=False)
    semantic = Path(args.admission_control).resolve()
    actual = Path(args.actual_export_control).resolve()
    admission(semantic, actual)
    sources = {p: digest(ROOT / p) for p in SOURCES}
    final_protocol = final / 'protocol.json'
    binding = dict(final_test_protocol_sha256=digest(final_protocol),
                   semantic_control_sha256=digest(semantic), actual_control_sha256=digest(actual),
                   actual_archive_binding_sha256=(digest(actual.parent / 'archive_storage_binding.json')
                     if (actual.parent / 'archive_storage_binding.json').exists() else None))
    protocol = dict(status='WAIT_FINAL_VALIDATION_FIXED_MODEL_THEN_EXPORT_ALL_HELD_ROLES',
        source_commit_at_initialization=git_head(ROOT), sources=sources,
        final_test_name=final.name, final_test_pid=args.final_test_pid, bindings=binding,
        role_exports=['MODEL_SELECTION_ONLY', 'PROTECTED_DIAGNOSTIC_ONLY', 'FINAL_TEST_ONLY'],
        reference_geometry_reads=0, fit_calls=0, model_selection_changed=False, full_contract_pass=False)
    atomic(out / 'protocol.json', protocol)
    process = Path('/proc') / str(args.final_test_pid)
    initial_ticks = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]

    def verify():
        if sources != {p: digest(ROOT / p) for p in sources} or binding != dict(
            final_test_protocol_sha256=digest(final_protocol),
            semantic_control_sha256=digest(semantic), actual_control_sha256=digest(actual),
            actual_archive_binding_sha256=(digest(actual.parent / 'archive_storage_binding.json')
              if (actual.parent / 'archive_storage_binding.json').exists() else None)):
            raise ValueError('Bound final export sources or admission changed')

    try:
        while not (final / 'receipt.json').exists():
            verify()
            if (final / 'failure.json').exists():
                raise RuntimeError('The existing final TEST controller actually failed')
            try:
                stat = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()
                argv = [v.decode() for v in process.joinpath('cmdline').read_bytes().split(b'\0') if v]
            except FileNotFoundError:
                if (final / 'receipt.json').exists():
                    break
                raise RuntimeError('Final TEST process exited without its completion receipt')
            if stat[19] != initial_ticks or stat[0] == 'Z' or final.name not in argv or 'tools.continue_validation_fixed_test' not in argv:
                raise RuntimeError('Final TEST live process binding changed without completion')
            atomic(out / 'progress.json', dict(status='WAITING_EXISTING_LIVE_COMPLETE_FINAL_TEST',
                controller_pid=os.getpid(), final_test_pid=args.final_test_pid,
                final_test_start_ticks=initial_ticks, reference_geometry_reads=0, fit_calls=0, unix_time=time.time()))
            time.sleep(args.poll_seconds)
        verify()
        completion = read(final / 'receipt.json')
        selection = final / 'final_selection.json'; selected = read(selection)
        if (completion['status'] != 'ALL_REGISTERED_FINAL_TEST_GEOMETRY_AND_FIELDS_EVALUATED_NOT_PROMOTED'
                or completion['protocol_sha256'] != digest(final_protocol)
                or completion['final_selection_sha256'] != digest(selection)
                or completion['fit_calls'] != 0 or completion['test_evaluations_used_for_selection'] != 0):
            raise ValueError('Authentic unchanged validation-fixed final TEST required')
        continuation = BASE / name(selected['selected_continuation'])
        protected_suffix = dict(held_stages())['PROTECTED_DIAGNOSTIC_ONLY']
        runs = [('MODEL_SELECTION_ONLY', ROOT / selected['validation_summary_path'], None),
                ('PROTECTED_DIAGNOSTIC_ONLY', continuation.with_name(continuation.name + '_' + protected_suffix) / 'summary.json', None),
                ('FINAL_TEST_ONLY', ROOT / completion['geometry_summary_path'], selection)]
        exported = []
        for role, summary_path, fixed in runs:
            summary_path = summary_path.resolve(); summary_path.relative_to(ROOT)
            summary = read(summary_path)
            if (summary['query_checkpoint_sha256'] != selected['query_checkpoint_sha256']
                    or summary['selector_checkpoint_sha256'] != selected['selector_checkpoint_sha256']
                    or summary['role'] != role):
                raise ValueError('Role output belongs to another final model pair')
            export_name = out.name + '_' + role.lower()
            args_child = ['--prediction-run', str(summary_path.parent), '--experiment-name', export_name, '--role', role]
            if fixed is not None:
                args_child += ['--final-selection', str(fixed)]
            verify()
            with (out / (role.lower() + '.log')).open('a') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'tools.export_full_weld_predictions', *args_child],
                                         cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                atomic(out / 'progress.json', dict(status='EXPORTING_AUTHENTIC_FINAL_MODEL_ROLE',
                    controller_pid=os.getpid(), child_pid=child.pid, role=role, reference_geometry_reads=0, fit_calls=0))
                if child.wait() != 0:
                    raise RuntimeError('Final CSV export failed for role ' + role)
            receipt = BASE / export_name / 'receipt.json'; result = read(receipt)
            if result['role'] != role or result['reference_geometry_reads'] != 0 or result['fit_calls'] != 0:
                raise ValueError('Final output role export differs')
            exported.append(dict(role=role, path=str(receipt.relative_to(ROOT)), sha256=digest(receipt),
                                 exported_child_rows=result['exported_child_rows']))
        verify()
        atomic(out / 'receipt.json', dict(status='ALL_THREE_HELD_ROLES_FINAL_MODEL_CSV_EXPORTED_NOT_CERTIFIED',
            protocol_sha256=digest(out / 'protocol.json'), final_selection_sha256=digest(selection),
            query_checkpoint_sha256=selected['query_checkpoint_sha256'],
            selector_checkpoint_sha256=selected['selector_checkpoint_sha256'], roles=exported,
            reference_geometry_reads=0, fit_calls=0, original_predictions_changed=False,
            business_frames_customer_layout_and_model_quality_not_certified=True, full_contract_pass=False))
    except BaseException as exc:
        atomic(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--final-test-name', required=True)
    parser.add_argument('--final-test-pid', required=True, type=int)
    parser.add_argument('--admission-control', required=True)
    parser.add_argument('--actual-export-control', required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    main(parser.parse_args())
