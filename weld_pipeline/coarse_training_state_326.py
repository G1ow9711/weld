"""Complete finite state with explicitly registered partial optimizer binding."""
import math
import torch
from .source_checkpoint_232 import tree_hash, tensor_layout, plain

GEOMETRY_PREFIXES = ("point_encoder.", "token_encoder.", "query_positions.",
    "parent_queries_embedding.", "parent_decoder.", "line_endpoints.",
    "circle_parameters.", "sequence_positions.", "sequence_encoder.", "sequence_interval.")
DISCRETE_PREFIXES = ("presence.", "parent_mode.", "parent_closed.", "parent_direction.",
    "parent_turns.", "sequence_kind.", "sequence_weld_type.", "sequence_gap.")


def set_registered_stage(model, stage):
    assert stage in ("A_FIXED_COARSE_GEOMETRY", "B_DYNAMIC_COARSE_PUBLICATION")
    prefixes = GEOMETRY_PREFIXES + (DISCRETE_PREFIXES if stage.startswith("B_") else ())
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        parameter.grad = None
    return [p for p in model.parameters() if p.requires_grad]


def finite_stage_state(model, optimizer):
    state, opt = model.state_dict(), optimizer.state_dict()
    checked = 0
    def check(value):
        nonlocal checked
        if isinstance(value, torch.Tensor):
            checked += 1
            assert torch.isfinite(value).all(), "Nonfinite complete model/optimizer tensor"
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
        elif isinstance(value, float):
            assert math.isfinite(value)
    check(state)
    check(opt)
    names = {id(p): n for n, p in model.named_parameters()}
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    observed, groups = [], []
    for index, group in enumerate(optimizer.param_groups):
        ordered = []
        for parameter in group["params"]:
            assert parameter.requires_grad and id(parameter) in names
            name = names[id(parameter)]
            assert name not in observed
            observed.append(name)
            ordered.append(dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype), device=str(parameter.device)))
        groups.append(dict(group_index=index, parameters=ordered,
            actual_options=plain({k: v for k, v in group.items() if k != "params"})))
    assert set(observed) == set(trainable)
    assert not set(observed).intersection(frozen)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    return dict(finite=True, every_model_and_optimizer_tensor_checked=True, tensor_count=checked,
        model_content_sha256=tree_hash(state), optimizer_content_sha256=tree_hash(opt),
        parameter_binding=dict(all_registered_trainable_parameters_bound_once=True,
            frozen_parameters_excluded_from_optimizer=True, trainable_names=trainable, frozen_names=frozen,
            parameter_groups=groups), optimizer_tensor_devices=tensor_layout(opt))


def frozen_parameter_hash(model):
    return tree_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})
