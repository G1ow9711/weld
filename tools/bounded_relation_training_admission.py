"""Require actual full-job optimizer parity before using bounded TRAIN reuse."""
from pathlib import Path

from tools.bounded_relation_training_backend import configure as numerical_configure
from weld_pipeline.sjg_local_runtime_904 import read,digest

ROOT=Path(__file__).resolve().parents[1]


def configure(reference_controls,synthetic_control,geometry_control,competition_control,fit_control,output):
    path=Path(fit_control).resolve();path.relative_to(ROOT)
    proof=read(path)
    if (proof['status']!='PASS_BOUNDED_ORIGINAL_FULL_JOB_AND_ACTUAL_EIGHT_STEP_FIT'
            or proof['actual_optimizer_updates_per_model']!=8 or proof['held_reference_reads']!=0
            or not proof['all_input_tokens_and_costs_and_views_bitwise_equal']
            or not proof['all_eight_histories_equal']
            or not proof['final_parameter_and_optimizer_tensors_bitwise_equal']
            or not proof['final_unary_penalties_and_selection_bitwise_equal']
            or not proof['forged_bounded_source_binding_rejected']
            or proof['sources']!={n:digest(ROOT/n) for n in proof['sources']}):
        raise ValueError('Current actual full-job and eight-step optimizer parity is required')
    loader,guards,metadata=numerical_configure(reference_controls,synthetic_control,
        geometry_control,competition_control,output)
    if proof['backend']!=metadata:
        raise ValueError('Actual optimizer parity belongs to another bounded backend recipe')
    guards=list(dict.fromkeys([*guards,*proof['sources'],
        'tools/bounded_relation_training_admission.py',str(path.relative_to(ROOT))]))
    metadata=dict(metadata,actual_fit_control_path=str(path.relative_to(ROOT)),
        actual_fit_control_sha256=digest(path),actual_fit_optimizer_updates_per_model=8)
    return loader,guards,metadata


def fit_arguments(parser):
    parser.add_argument('--bounded-training-control',required=True)
