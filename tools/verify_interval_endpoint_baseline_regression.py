"""Preserve every original ownership/scoring check and reject real partial V5.

The partial checkpoint's actual receipt is used. Rejection must come from its
diagnostic status, rather than an absent invented checkpoint receipt.
"""
import argparse
import ast
import inspect

from tools import verify_local_baseline_regression as original
from tools.run_interval_endpoint_baseline_stage import regression_sources
from tools.run_interval_endpoint_full_dataset_stage import configure
from weld_pipeline.server_interval_endpoint_checkpoint import verify_complete_interval_endpoint_fit


class ActualReceipt(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Constant(self, node):
        if node.value == 'checkpoint_receipt.json':
            self.count += 1
            return ast.copy_location(ast.Constant('receipt.json'), node)
        return node


def main(args):
    tree = ast.parse(inspect.getsource(original.main))
    change = ActualReceipt(); tree = change.visit(tree)
    if change.count != 1:
        raise ValueError('Original partial checkpoint receipt structure changed')
    ast.fix_missing_locations(tree)
    namespace = dict(original.main.__globals__, __file__=__file__,
        configure=configure, regression_sources=regression_sources,
        verify_complete_local_query_fit=verify_complete_interval_endpoint_fit)
    exec(compile(tree, __file__, 'exec'), namespace)
    return namespace['main'](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='full_dataset_v2')
    parser.add_argument('--baseline-manifest', required=True)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--query-training-name', required=True)
    main(parser.parse_args())
