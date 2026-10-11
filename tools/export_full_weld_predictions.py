"""Export authenticated complete model outputs without reading label geometry."""
import argparse
from pathlib import Path
import re
import traceback

from tools.assess_full_weld_output_fields import (
    local, select_records, authenticate, pool_freeze_fingerprint,
)
from weld_pipeline.server_full_shard_dataset import FullShardDataset
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset, ROLES
from weld_pipeline.server_predicted_parent_pool import decoded_subset
from weld_pipeline.server_parent_prediction_runtime import load_gzip
from weld_pipeline.server_full_weld_csv_export import export_decoded, verify_roundtrip
from weld_pipeline.sjg_local_runtime_904 import read, digest, write
from tools.record_training_experiment import git_head

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
SOURCES = ['tools/export_full_weld_predictions.py', 'weld_pipeline/server_full_weld_csv_export.py',
    'tools/assess_full_weld_output_fields.py', 'weld_pipeline/runtime_csv_export_287.py',
    'weld_pipeline/server_full_shard_dataset.py', 'weld_pipeline/server_full_evaluation_dataset.py',
    'weld_pipeline/server_predicted_parent_pool.py', 'weld_pipeline/server_parent_prediction_runtime.py',
    'weld_pipeline/server_full_prediction_runtime.py', 'tools/evaluate_full_weld_predictions.py']


def main(args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.experiment_name):
        raise ValueError('Plain independent local export experiment required')
    run = local(ROOT / args.prediction_run, directory=True)
    dataset = FullShardDataset(ROOT) if args.role == 'TRAIN' else EvaluationShardDataset(ROOT)
    paths, summary = select_records(run, dataset, args.role, args.final_selection)
    # The existing role verifier authenticates every required held view and
    # requires a prior validation-fixed model for TEST. No reference() or
    # supervision() call is made by this exporter.
    protocol_sha = digest(run / 'protocol.json')
    sources = {name: digest(ROOT / name) for name in SOURCES}
    out = BASE / args.experiment_name; out.mkdir(exist_ok=False)
    bindings = [dict(path=str(path.relative_to(ROOT)), sha256=digest(path)) for path in paths]
    protocol = dict(status='AUTHENTICATED_COMPLETE_MODEL_CSV_EXPORT', role=args.role,
        source_commit_at_initialization=git_head(ROOT), sources=sources,
        prediction_run=str(run.relative_to(ROOT)), prediction_protocol_sha256=protocol_sha,
        registry_sha256=dataset.registry_sha256, records=bindings,
        prior_complete_role_summary_sha256=digest(summary) if summary else None,
        reference_geometry_reads=0, fit_calls=0, predictions_changed=False,
        TRAIN_snapshots_not_one_fixed_model=args.role == 'TRAIN', full_contract_pass=False)
    write(out / 'protocol.json', protocol)
    results = []
    try:
        for index, (path, binding) in enumerate(zip(paths, bindings)):
            if digest(path) != binding['sha256'] or digest(run / 'protocol.json') != protocol_sha:
                raise ValueError('Authenticated source record changed')
            receipt, freeze = authenticate(run, path, protocol_sha, dataset.registry_sha256, args.role)
            parents = load_gzip(path.parent / 'selected_parents.json.gz')
            decoded = decoded_subset(parents, range(len(parents)))
            folder = out / f'job_{index:04d}'
            schema = export_decoded(folder, receipt['job_key'], decoded)
            check = verify_roundtrip(folder, decoded)
            if sources != {name: digest(ROOT / name) for name in sources}:
                raise ValueError('Export source changed')
            files = {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}
            result = dict(source_record=binding, job_key=receipt['job_key'], role=args.role,
                query_checkpoint_sha256=read(run / 'protocol.json')['query_checkpoint_sha256'],
                selector_checkpoint_sha256=freeze['checkpoint_sha256'],
                source_geometry_sha256=freeze['selected_geometry_sha256'],
                rows=schema['published_rows'], parents=schema['published_parents'],
                roundtrip=check, artifacts=files, reference_geometry_reads=0, fit_calls=0,
                complete_business_attributes_or_formal_customer_layout_certified=False)
            write(folder / 'receipt.json', result)
            results.append(dict(path=str((folder / 'receipt.json').relative_to(ROOT)),
                                sha256=digest(folder / 'receipt.json'), rows=result['rows']))
            print('AUTHENTICATED_FULL_OUTPUT_CSV', index, result['rows'], flush=True)
        write(out / 'receipt.json', dict(status='ALL_AUTHENTICATED_FROZEN_OUTPUTS_CSV_EXPORTED_NOT_CERTIFIED',
            role=args.role, protocol_sha256=digest(out / 'protocol.json'), records=results,
            exported_records=len(results), exported_child_rows=sum(item['rows'] for item in results),
            every_original_parent_and_child_and_signed_frame_preserved=True,
            reference_geometry_reads=0, fit_calls=0, TRAIN_snapshots_not_one_fixed_model=args.role == 'TRAIN',
            original_geometry_predictions_unchanged=True, full_contract_pass=False))
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-run', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--role', choices=['TRAIN', *ROLES], required=True)
    parser.add_argument('--final-selection')
    main(parser.parse_args())
