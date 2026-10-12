"""Compare the same complete endpoint model against the original best cases."""
import argparse
from types import FunctionType

from tools import run_local_baseline_stage as original
from tools.run_interval_endpoint_full_dataset_stage import configure, SOURCES as PIPELINE_SOURCES
from weld_pipeline.server_interval_endpoint_checkpoint import load_interval_endpoint_model


EXTRA_SOURCES = list(dict.fromkeys([*original.EXTRA_SOURCES, *PIPELINE_SOURCES,
    'tools/run_interval_endpoint_baseline_stage.py',
    'tools/verify_interval_endpoint_baseline_regression.py']))


def regression_sources():
    namespace = dict(original.regression_sources.__globals__, EXTRA_SOURCES=EXTRA_SOURCES)
    function = FunctionType(original.regression_sources.__code__, namespace,
        original.regression_sources.__name__, original.regression_sources.__defaults__,
        original.regression_sources.__closure__)
    return function()


def main(args):
    namespace = dict(original.main.__globals__, configure=configure,
        load_local_query_model=load_interval_endpoint_model, EXTRA_SOURCES=EXTRA_SOURCES)
    execute = FunctionType(original.main.__code__, namespace, original.main.__name__,
                           original.main.__defaults__, original.main.__closure__)
    return execute(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    main(parser.parse_args())
