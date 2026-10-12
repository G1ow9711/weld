"""Preserve complete parent/held stages with the explicit endpoint architecture.

The existing raw XYZ scorer and proved disk relation backend execute unchanged.
Private function namespaces declare only the new factory, complete-fit verifier,
source bindings and child dispatch. Existing model controllers stay untouched.
"""
import argparse
from pathlib import Path
import subprocess
from types import FunctionType, SimpleNamespace

from tools import run_local_full_dataset_stage as local
from tools import run_disk_cached_local_full_dataset_stage as disk
from weld_pipeline.server_interval_endpoint_checkpoint import (
    load_interval_endpoint_model, verify_complete_interval_endpoint_fit)
from weld_pipeline.server_interval_endpoint_residual_model import ARCHITECTURE


ROOT = Path(__file__).resolve().parents[1]
SOURCES = list(dict.fromkeys([*disk.SOURCES,
    'tools/run_interval_endpoint_full_dataset_stage.py',
    'weld_pipeline/server_interval_endpoint_checkpoint.py',
    'weld_pipeline/server_interval_endpoint_residual_model.py',
    'tools/train_full_interval_endpoint_geometry.py',
    'tools/compare_interval_endpoint_residual.py']))


class EndpointSubprocess:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def arguments(self, argv):
        result = local.child_arguments(argv, self.dataset_name)
        if '-m' in result:
            index = result.index('-m') + 1
            if result[index] == 'tools.run_local_full_dataset_stage':
                result[index] = 'tools.run_interval_endpoint_full_dataset_stage'
        return result

    def Popen(self, argv, *args, **kwargs):
        return subprocess.Popen(self.arguments(argv), *args, **kwargs)

    def run(self, argv, *args, **kwargs):
        return subprocess.run(self.arguments(argv), *args, **kwargs)


def configure(dataset_name):
    # The closure that records raw XYZ input bindings also declares the new
    # architecture. Updating only the outer factory would leave a stale tag.
    namespace = dict(local.configure.__globals__, ARCHITECTURE=ARCHITECTURE,
        load_local_query_model=load_interval_endpoint_model,
        verify_complete_local_query_fit=verify_complete_interval_endpoint_fit,
        SOURCES=SOURCES)
    localized = FunctionType(local.configure.__code__, namespace, local.configure.__name__,
                             local.configure.__defaults__, local.configure.__closure__)
    namespace = dict(disk.configure.__globals__, original=SimpleNamespace(configure=localized))
    configured = FunctionType(disk.configure.__code__, namespace, disk.configure.__name__,
                              disk.configure.__defaults__, disk.configure.__closure__)
    base, train, held, modules = configured(dataset_name)
    bridge = EndpointSubprocess(dataset_name)
    for module in modules.values():
        if hasattr(module, 'subprocess'):
            module.subprocess = bridge
    return base, train, held, modules


def main(args):
    namespace = dict(local.main.__globals__, configure=configure,
                     SOURCES=SOURCES, ARCHITECTURE=ARCHITECTURE)
    execute = FunctionType(local.main.__code__, namespace, local.main.__name__,
                           local.main.__defaults__, local.main.__closure__)
    return execute(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    main(parser.parse_args())
