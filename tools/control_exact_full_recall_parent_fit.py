"""Exercise the entire frozen actual TRAIN pool with verified exact costs."""
import argparse
from pathlib import Path

from tools.control_full_recall_parent_fit import main
from tools.exact_reference_training_backend import configure, configured_main, control_arguments

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--pool-name',required=True)
    parser.add_argument('--producer-pid',type=int,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--query-sha256',required=True)
    parser.add_argument('--selector-checkpoint',required=True)
    parser.add_argument('--selector-sha256',required=True)
    parser.add_argument('--steps',type=int,default=8)
    control_arguments(parser)
    args = parser.parse_args()
    if Path(args.experiment_name).name != args.experiment_name:
        raise ValueError('An independent local control name is required')
    output = ROOT/'server_rebuild/full_dataset_v1'/args.experiment_name
    loader, guards, metadata = configure(args.reference_control,args.synthetic_control,output)
    guards.append('tools/control_full_recall_parent_fit.py')
    configured_main(main,loader,guards,metadata,__file__,output)(args)
