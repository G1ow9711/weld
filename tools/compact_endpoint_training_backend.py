"""Bind verified exact reference compilation to unchanged TRAIN orchestration.

The original job loader, tokens, priors, complete-set objective, optimizer and
prediction code run unchanged. Only its offline compact reference compiler is
replaced, using a private function namespace rather than mutating any module
used by an already running process. Provenance binds the actual comparison
receipts, every result and the current numerical implementation.
"""
from pathlib import Path
from types import FunctionType

from weld_pipeline.server_compact_endpoint_reference import compile_compact_endpoint_reference
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools.train_full_predicted_parent_selector import load_job

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SOURCES = ['tools/compact_endpoint_training_backend.py',
                   'weld_pipeline/server_compact_endpoint_reference.py']


def owned(path):
    path = Path(path).resolve()
    path.relative_to(ROOT)
    return path


def configure(reference_controls, synthetic_control, report_directory):
    helper_sha = digest(ROOT/'weld_pipeline/server_compact_endpoint_reference.py')
    synthetic_path = owned(synthetic_control)
    synthetic = read(synthetic_path)
    if (synthetic['status'] != 'PASS_EVERY_ORIGINAL_COST_AND_GLOBAL_MATCHING_SYNTHETIC_CONTROL_NOT_MODEL_QUALITY'
            or synthetic['helper_sha256'] != helper_sha
            or synthetic['source_sha256'] != digest(ROOT/'tools/verify_compact_endpoint_reference.py')
            or synthetic['original_compiler_sha256'] != digest(ROOT/'weld_pipeline/server_offline_parent_set_reference.py')
            or not synthetic['all_costs_and_tp_bitwise_equal']
            or not synthetic['partial_parent_later_child_preserved']
            or not synthetic['closed_circle_anchor_not_rejected']
            or not synthetic['malformed_ownership_rejected']
            or not synthetic['compact_sampled_endpoints_bitwise_equal']
            or not synthetic['unavailable_compact_row_unsampled_and_FP_retained']
            or not synthetic['original_runtime_decoder_required_for_compact_rows']
            or not synthetic['original_sampler_ignores_stored_points_preserved']
            or synthetic['reused_original_control_source_sha256']
                != digest(ROOT/'tools/verify_exact_endpoint_reference.py')):
        raise ValueError('Current exact compiler needs its actual synthetic semantic controls')
    bound = [synthetic_path]
    controls = []
    registry_sha = digest(ROOT/'server_rebuild/full_dataset_v1/training_registry.json')
    for value in reference_controls:
        folder = owned(value)
        receipt_path, protocol_path = folder/'receipt.json', folder/'protocol.json'
        receipt, protocol = read(receipt_path), read(protocol_path)
        if (receipt['status'] != 'PASS_ALL_ACTUAL_ORIGINAL_COSTS_AND_GLOBAL_MATCHING_NOT_MODEL_ACCURACY'
                or not receipt['all_costs_and_tp_bitwise_equal'] or receipt['fit_calls'] != 0
                or receipt['held_reference_reads'] != 0 or receipt['predictions_changed']
                or receipt['protocol_sha256'] != digest(protocol_path)
                or protocol['registry_sha256'] != registry_sha
                or protocol['sources'] != {name:digest(ROOT/name) for name in protocol['sources']}):
            raise ValueError('An actual complete-cost comparison belongs to different inputs or source')
        bound.extend([receipt_path, protocol_path])
        for record in receipt['records']:
            path = owned(ROOT/record['path'])
            path.relative_to(folder)
            if digest(path) != record['sha256']:
                raise ValueError('An actual full-cost comparison result changed')
            for view in read(path)['views']:
                if not view['all_costs_and_tp_bitwise_equal'] or not view['global_assignment_indices_equal']:
                    raise ValueError('Every cost and the global matching must agree')
            bound.append(path)
        controls.append(dict(directory=str(folder.relative_to(ROOT)),
            receipt_sha256=digest(receipt_path),complete_pool=receipt['complete_pool_control'],
            actual_records=receipt['actual_input_records']))
    if not any(c['complete_pool'] for c in controls) or not any(not c['complete_pool'] for c in controls):
        raise ValueError('Actual whole-pool and saved-output controls are both required')
    guards = [*BACKEND_SOURCES, 'tools/verify_exact_endpoint_reference.py',
              'tools/verify_compact_endpoint_reference.py',
              'tools/verify_actual_compact_endpoint_reference.py',
              *[str(p.relative_to(ROOT)) for p in bound]]
    metadata = dict(backend='EXACT_ORIGINAL_D3_COMPACT_ENDPOINT_PRE_SAMPLING',
        helper_sha256=helper_sha,synthetic_control=str(synthetic_path.relative_to(ROOT)),
        synthetic_control_sha256=digest(synthetic_path),actual_controls=controls,
        prediction_rows_or_label_columns_removed=0,all_original_costs_and_global_assignment_retained=True,
        original_training_loader_sha256=digest(ROOT/'tools/train_full_predicted_parent_selector.py'),
        sampling_skipped_only_for_proven_unavailable_pairs=True,optimizer_or_inference_changed=False)
    report_directory = owned(report_directory)
    def compiler(parents, target, decode_subset, batch_size=1024):
        def progress(value):
            path = report_directory/'reference_compilation_progress.json'
            temporary = path.with_name(path.name+'.incomplete')
            write(temporary,dict(status='ACTUAL_EXACT_ALL_PARENT_REFERENCE_COMPILATION',
                reference_parents=len(target['parents']),reference_children=target['original_children'],**value))
            temporary.replace(path)
        return compile_compact_endpoint_reference(parents,target,decode_subset,batch_size,progress=progress)
    environment = dict(load_job.__globals__)
    environment['compile_compact_reference_table'] = compiler
    loader = FunctionType(load_job.__code__,environment,load_job.__name__,
                          load_job.__defaults__,load_job.__closure__)
    return loader, list(dict.fromkeys(guards)), metadata


def configured_main(original, loader, guards, metadata, entry_file, output):
    """Use an independent namespace; all bound original module globals survive."""
    environment = dict(original.__globals__)
    environment['load_job'] = loader
    environment['SOURCE_NAMES'] = list(dict.fromkeys([*environment['SOURCE_NAMES'],
        *guards,str(owned(entry_file).relative_to(ROOT))]))
    environment['__file__'] = str(owned(entry_file))
    original_write = environment['write']
    output = owned(output)
    def bound_write(path, value):
        path = Path(path)
        if path == output/'protocol.json':
            binding = output/'reference_backend_binding.json'
            if binding.exists():
                if read(binding) != metadata:
                    raise ValueError('Independent reference backend binding changed')
            else:
                original_write(binding,metadata)
        return original_write(path,value)
    environment['write'] = bound_write
    return FunctionType(original.__code__,environment,original.__name__,
                        original.__defaults__,original.__closure__)


def control_arguments(parser):
    parser.add_argument('--reference-control',action='append',required=True)
    parser.add_argument('--synthetic-control',required=True)
