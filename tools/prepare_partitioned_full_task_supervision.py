"""Execute unchanged complete owner-witness preparation in a declared partition."""
import argparse
from pathlib import Path

from tools import prepare_full_task_supervision as original
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


def main(args):
    if Path(args.dataset_name).name != args.dataset_name or args.dataset_name in ('', '.', '..', 'full_dataset_v1'):
        raise ValueError('An independent repaired dataset directory is required')
    root = original.ROOT
    base = root/'server_rebuild'/args.dataset_name
    proof = read(base/'registration_repair/receipt.json')
    if proof['bindings'] != {name:digest(base/name) for name in proof['bindings']}:
        raise ValueError('Corrected complete registration changed')
    write(base/'partitioned_supervision_execution_binding.json',dict(
        dataset_directory=str(base.relative_to(root)), registration_proof_sha256=digest(base/'registration_repair/receipt.json'),
        sources={name:digest(root/name) for name in ['tools/prepare_partitioned_full_task_supervision.py',
            'tools/prepare_full_task_supervision.py','weld_pipeline/server_partitioned_full_dataset.py']},
        witness_algorithm_modified=False, reference_geometry_or_raw_frames_modified=False,
        model_fit_calls=0, held_supervision_admitted=False))
    original.BASE = base
    original.DEST = base/'full_task_supervision_v1'
    original.FullShardDataset = lambda value: PartitionedFullShardDataset(value,base)
    original.main()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-name',required=True);main(parser.parse_args())
