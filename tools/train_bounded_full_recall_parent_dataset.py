"""Fit all TRAIN parent sets with unchanged objectives and bounded reuse."""
import argparse
from pathlib import Path

from tools.train_full_recall_parent_dataset import main
from tools.compact_endpoint_training_backend import control_arguments
from tools.bounded_relation_training_backend import configured_main,relation_arguments
from tools.bounded_relation_training_admission import configure,fit_arguments

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--passes',type=int,default=3)
    parser.add_argument('--steps-per-job',type=int,default=8)
    parser.add_argument('--selector-checkpoint')
    parser.add_argument('--selector-sha256')
    parser.add_argument('--query-checkpoint',required=True)
    parser.add_argument('--query-sha256',required=True)
    parser.add_argument('--query-training-receipt')
    parser.add_argument('--allow-partial-query',action='store_true')
    control_arguments(parser);relation_arguments(parser);fit_arguments(parser)
    args=parser.parse_args()
    if not args.experiment_name or Path(args.experiment_name).name!=args.experiment_name:
        raise ValueError('An independent local experiment name required')
    output=ROOT/'server_rebuild/full_dataset_v1'/args.experiment_name
    loader,guards,metadata=configure(args.reference_control,args.synthetic_control,
        args.bounded_geometry_control,args.bounded_competition_control,args.bounded_training_control,output)
    configured_main(main,loader,guards,metadata,__file__,output)(args.experiment_name,
        args.passes,args.steps_per_job,args.selector_checkpoint,args.selector_sha256,
        args.query_checkpoint,args.query_sha256,args.query_training_receipt,args.allow_partial_query)
