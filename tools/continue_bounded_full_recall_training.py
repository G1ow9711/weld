"""Continue all query/parent fitting and held assessment with bounded reuse."""
import argparse
from pathlib import Path
from types import FunctionType
import time
import traceback

from tools.continue_full_recall_training import main,run_child,atomic_json,checked_name
from tools.compact_endpoint_training_backend import control_arguments
from tools.bounded_relation_training_backend import relation_arguments
from tools.bounded_relation_training_admission import configure,fit_arguments

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/full_dataset_v1'


def configured_continuation(args):
    parent_name=checked_name(args.parent_training_name)
    _,guards,metadata=configure(args.reference_control,args.synthetic_control,
        args.bounded_geometry_control,args.bounded_competition_control,args.bounded_training_control,BASE/parent_name)
    environment=dict(main.__globals__)
    environment['SOURCES']=list(dict.fromkeys([*environment['SOURCES'],*guards,
        'tools/continue_bounded_full_recall_training.py','tools/train_bounded_full_recall_parent_dataset.py']))
    environment['__file__']=__file__
    launcher=FunctionType(run_child.__code__,environment,run_child.__name__,run_child.__defaults__,run_child.__closure__)
    def dispatch(out,protocol,module,arguments,stage):
        if module=='tools.train_full_recall_parent_dataset':
            module='tools.train_bounded_full_recall_parent_dataset'
            arguments=list(arguments)+['--synthetic-control',args.synthetic_control,
                '--bounded-geometry-control',args.bounded_geometry_control,
                '--bounded-competition-control',args.bounded_competition_control,
                '--bounded-training-control',args.bounded_training_control]
            for value in args.reference_control:arguments+=['--reference-control',value]
        return launcher(out,protocol,module,arguments,stage)
    environment['run_child']=dispatch
    continuation=FunctionType(main.__code__,environment,main.__name__,main.__defaults__,main.__closure__)
    return continuation,metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--query-training-name',default='full_weld_reference_recall_v3')
    parser.add_argument('--parent-training-name',required=True)
    parser.add_argument('--passes',type=int,default=3)
    parser.add_argument('--steps-per-job',type=int,default=8)
    parser.add_argument('--selector-checkpoint')
    parser.add_argument('--selector-sha256')
    control_arguments(parser);relation_arguments(parser);fit_arguments(parser)
    args=parser.parse_args()
    try:
        if min(args.passes,args.steps_per_job)<1:raise ValueError('Complete positive visit budgets required')
        continuation,metadata=configured_continuation(args)
        out=BASE/checked_name(args.experiment_name);out.mkdir(exist_ok=True)
        binding=out/'reference_backend_binding.json'
        if binding.exists():
            from weld_pipeline.sjg_local_runtime_904 import read
            if read(binding)!=metadata:raise ValueError('Bounded continuation backend changed')
        else:atomic_json(binding,metadata)
        continuation(args)
    except BaseException:
        out=BASE/checked_name(args.experiment_name);out.mkdir(exist_ok=True)
        atomic_json(out/'failure.json',dict(status='INCOMPLETE_BOUNDED_CONTINUATION_NOT_PROMOTED',
            traceback=traceback.format_exc(),unix_time=time.time()))
        raise
