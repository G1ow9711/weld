"""Execute preserved original-best regression with explicit local-XYZ query context.

The original collection, final-model binding, original D3 scorer and historical
reference comparison are unchanged. Query loading/input-only generation and
their extra source fingerprints are declared in this separate process.
"""
import argparse
import importlib
from pathlib import Path
import sys
from types import FunctionType

from tools.run_local_full_dataset_stage import configure, SOURCES as LOCAL_SOURCES
from tools.run_partitioned_full_dataset_stage import entry_body
from tools.export_full_dataset_split_view import local_name
from weld_pipeline.server_local_query_checkpoint import load_local_query_model
from weld_pipeline.sjg_local_runtime_904 import digest, read, write


ROOT = Path(__file__).resolve().parents[1]
REGRESSION = 'tools.continue_validation_fixed_baseline_regression'
COMPARISON = 'tools.continue_original_best_reference_comparison'
EXTRA_SOURCES = [*LOCAL_SOURCES, 'tools/run_local_baseline_stage.py', 'tools/verify_local_baseline_regression.py']


def regression_sources():
    module = importlib.import_module(REGRESSION)
    return list(dict.fromkeys([*module.SOURCES, *EXTRA_SOURCES]))


def main(args):
    local_name(args.dataset_name)
    if args.module not in (REGRESSION, COMPARISON):
        raise ValueError('Only the complete original-best regression and its reference comparison are allowed')
    module = importlib.import_module(args.module)
    source = Path(module.__file__)
    namespace = dict(module.__dict__)
    sources = list(dict.fromkeys([*module.SOURCES, *EXTRA_SOURCES]))
    environment = dict(module.main.__globals__, SOURCES=sources)
    if args.module == REGRESSION:
        environment.update(configure=configure, load_query_model=load_local_query_model)
    namespace['main'] = FunctionType(module.main.__code__, environment, module.main.__name__,
                                    module.main.__defaults__, module.main.__closure__)
    namespace['SOURCES'] = sources
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if '--dataset-name' in arguments:
        raise ValueError('Declare the one dataset only in the explicit launcher context')
    tree, _ = entry_body(source.read_text(), args.dataset_name)
    if '--experiment-name' not in arguments:
        raise ValueError('An independent regression experiment name is required')
    name = arguments[arguments.index('--experiment-name')+1]
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('The regression name must be a plain local directory name')
    base = ROOT / 'server_rebuild' / args.dataset_name
    path = base / 'local_baseline_execution_bindings' / (name + '.json')
    path.parent.mkdir(exist_ok=True)
    binding = dict(module=args.module, arguments=arguments, dataset_name=args.dataset_name,
        original_entry_sha256=digest(source), sources={item:digest(ROOT / item) for item in sources},
        original_inputs_references_denominators_and_scoring_preserved=True,
        historical_reference_results_not_used_for_model_selection=True)
    if path.exists() and read(path) != binding:
        raise ValueError('The declared original-best regression binding changed')
    if not path.exists():
        write(path, binding)
    original = sys.argv
    try:
        sys.argv = [str(source), '--dataset-name', args.dataset_name, *arguments]
        exec(compile(tree, str(source), 'exec'), namespace, namespace)
    finally:
        sys.argv = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--module', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    main(parser.parse_args())
