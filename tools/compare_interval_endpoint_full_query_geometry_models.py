"""Run the original complete TRAIN query comparison for the endpoint model."""
import argparse
from types import FunctionType

from tools import compare_cached_full_query_geometry_models as original
from weld_pipeline.server_interval_endpoint_checkpoint import (
    load_interval_endpoint_model, verify_complete_interval_endpoint_fit)


SOURCES = list(dict.fromkeys([*original.SOURCES,
    'tools/compare_interval_endpoint_full_query_geometry_models.py',
    'weld_pipeline/server_interval_endpoint_checkpoint.py',
    'weld_pipeline/server_interval_endpoint_residual_model.py',
    'tools/train_full_interval_endpoint_geometry.py']))


def configured_main():
    assessor = FunctionType(original.assess_arm.__code__,
        dict(original.assess_arm.__globals__, load_local_query_model=load_interval_endpoint_model),
        original.assess_arm.__name__, original.assess_arm.__defaults__, original.assess_arm.__closure__)
    namespace = dict(original.main.__globals__, SOURCES=SOURCES, assess_arm=assessor,
                     verify_complete_local_query_fit=verify_complete_interval_endpoint_fit)
    return FunctionType(original.main.__code__, namespace, original.main.__name__,
                        original.main.__defaults__, original.main.__closure__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--baseline-comparison-name', default='full_query_dense_v3_v4_geometry_comparison_20261012_v1')
    parser.add_argument('--new-training-name', required=True)
    parser.add_argument('--new-training-pid', type=int, required=True)
    parser.add_argument('--experiment-name', required=True)
    configured_main()(parser.parse_args())
