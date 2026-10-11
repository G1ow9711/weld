"""Authenticate every actually prepared held XYZ cache before feature stages."""
from pathlib import Path

import numpy as np

from tools.prepare_full_evaluation_case import DATA, OUT, ROOT, WORKSPACE
from weld_pipeline.server_full_evaluation_dataset import EvaluationShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


def main():
    dataset = EvaluationShardDataset(ROOT)
    rows = []
    for job in dataset.jobs.values():
        path = WORKSPACE / job['job_key'] / 'inputs' / job['runtime_key']
        marker = read(OUT / job['job_key'] / 'stages/input.json')
        receipt = read(path / 'receipt.json')
        if (marker['registry_sha256'] != dataset.registry_sha256 or marker['optimizer_updates'] != 0
                or receipt['registry_sha256'] != dataset.registry_sha256 or receipt['fit_allowed']
                or receipt['coordinate_transform_applied'] or not receipt['only_XYZ_used']
                or receipt['input_extraction_GT_reads'] != 0 or receipt['STEP_reads'] != 0
                or receipt['source_sha256'] != job['workpiece_sha256']
                or receipt['evaluation_roles'] != sorted({v['role'] for v in job['versions']})):
            raise ValueError('Held XYZ extraction changed its source, role or inference contract')
        xyz_file = path / 'xyz_mm.npy'
        if (digest(DATA / job['workpiece_source']) != job['workpiece_sha256']
                or digest(xyz_file) != receipt['xyz_sha256']):
            raise ValueError('Prepared held XYZ or its original PLY changed')
        xyz = np.load(xyz_file, mmap_mode='r', allow_pickle=False)
        if xyz.dtype != np.float64 or xyz.shape != (job['original_points'], 3) or not np.isfinite(xyz).all():
            raise ValueError('Held XYZ dtype, shape or finite values changed')
        rows.append(dict(job_key=job['job_key'], points=len(xyz), workpiece_sha256=job['workpiece_sha256'],
            xyz_sha256=receipt['xyz_sha256'], input_receipt_sha256=digest(path / 'receipt.json'),
            input_stage_receipt_sha256=digest(OUT / job['job_key'] / 'stages/input.json')))
        xyz._mmap.close()
    result = dict(status='PASS_ALL_HELD_RAW_XYZ_PREPARED_NOT_COMPLETE_FEATURES_OR_MODEL_ACCURACY',
        clouds=len(rows), label_versions=len(dataset.versions), points=sum(r['points'] for r in rows), rows=rows,
        registry_sha256=dataset.registry_sha256, role_version_counts=dataset.registry['role_version_counts'],
        unit='mm', coordinate_frame='workpiece_local', coordinates_transformed=False,
        GT_geometry_as_input=False, STEP_as_input=False, optimizer_updates=0, predictions_scored=0,
        complete_feature_shards=len(dataset.ready_jobs()), complete_feature_preparation_still_required=True,
        source_sha256=digest(__file__))
    destination = dataset.base / 'full_evaluation_xyz_validation_v1.json'
    if destination.exists():
        raise ValueError('Immutable held XYZ validation already exists')
    write(destination, result)
    print(result['status'], len(rows), result['points'], flush=True)


if __name__ == '__main__':
    main()
