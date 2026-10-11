"""Compare all original costs on authenticated actual TRAIN pools or outputs."""
import argparse
from types import FunctionType

from tools import verify_actual_endpoint_reference as original
from weld_pipeline.server_compact_endpoint_reference import compile_compact_endpoint_reference


def main(args):
    environment=dict(original.compare.__globals__)
    environment['compile_exact_endpoint_reference']=compile_compact_endpoint_reference
    compare=FunctionType(original.compare.__code__,environment,original.compare.__name__,
                         original.compare.__defaults__,original.compare.__closure__)
    environment=dict(original.main.__globals__)
    environment['compare']=compare
    environment['SOURCES']=list(dict.fromkeys([*original.SOURCES,
        'tools/verify_actual_compact_endpoint_reference.py',
        'weld_pipeline/server_compact_endpoint_reference.py',
        'weld_pipeline/server_full_prediction_runtime.py','weld_pipeline/world_set_runtime_285.py']))
    verify=FunctionType(original.main.__code__,environment,original.main.__name__,
                        original.main.__defaults__,original.main.__closure__)
    verify(args)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pool-name'); modes.add_argument('--assessment-run')
    parser.add_argument('--job'); parser.add_argument('--output-name',required=True)
    main(parser.parse_args())
