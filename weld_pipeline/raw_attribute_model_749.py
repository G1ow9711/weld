"""Frozen639-compatible raw attribute architecture and explicit named Adam migration."""
from copy import deepcopy
import torch
from weld_pipeline.reachable_physical_model_649 import ReachablePhysicalContainer
from weld_pipeline.ordered_raw_attribute_head_744 import RawBoundedAttributeHead,VERSION
from weld_pipeline.resume_setloss_optimizers_689 import _copy_state,_state_equal

class RawPhysicalContainer(ReachablePhysicalContainer):
    def __init__(self,original115,original_hybrid,config):
        super().__init__(original115,original_hybrid,config)
        assert config.get("raw_attribute_stage")==VERSION
        scopes=config["raw_attribute_statistics"]["scopes"]
        self.attributes=RawBoundedAttributeHead(self.attributes,scopes["local"])
        self.path_attributes=RawBoundedAttributeHead(self.path_attributes,scopes["path"])

def install_raw_adapters(model,optimizers,bindings,statistics,*,initialization_seed=20261007746):
    """Call after exact639 model/Adam resume. Existing Parameter objects survive."""
    assert "raw_attribute_stage" not in model.config
    assert statistics["source_count"]==65 and statistics["original_denominators"]["original_G"]==3593
    named_before=dict(model.named_parameters())
    slots={name:deepcopy(optimizers["attributes"].state[p]) for name,p in named_before.items()
        if p in optimizers["attributes"].state}
    old_names=list(bindings["attributes"])
    # New random initialization is deterministic and does not advance639 RNG.
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(initialization_seed)
        for prefix,name in (("local","attributes"),("path","path_attributes")):
            old=getattr(model,name);device=next(old.parameters()).device
            new=RawBoundedAttributeHead(old,statistics["scopes"][prefix]).to(device)
            new.train(old.training);setattr(model,name,new)
    named_after=dict(model.named_parameters())
    assert all(named_after[name] is parameter for name,parameter in named_before.items())
    opt=optimizers["attributes"]
    rates={"attributes_state_boolean":1e-3,"attributes_association":3e-4,"attributes_numeric_normal":3e-4}
    changed=[]
    for group in opt.param_groups:
        assert group["phase_name"] in rates
        changed.append(dict(phase_name=group["phase_name"],old_lr=group["lr"],new_lr=rates[group["phase_name"]]))
        group["lr"]=rates[group["phase_name"]]
        assert group["weight_decay"]==1e-4
    new_names=[]
    for prefix in ("attributes","path_attributes"):
        names=[n for n in named_after if n.startswith(prefix+".raw_adapter.")]
        assert names and not any(n in named_before for n in names)
        opt.add_param_group(dict(params=[named_after[n] for n in names],lr=3e-4,weight_decay=1e-4,
            phase_name=prefix+"_raw_bypass"))
        new_names.extend(names)
    bindings["attributes"]=old_names+new_names
    assert [id(p) for group in opt.param_groups for p in group["params"]]==[id(named_after[n]) for n in bindings["attributes"]]
    assert all(n not in named_before and named_after[n] not in opt.state for n in new_names)
    assert len(opt.state)==len(slots)
    for name,slot in slots.items():assert _state_equal(slot,opt.state[named_after[name]]),name
    model.config.update(raw_attribute_stage=VERSION,raw_attribute_statistics=deepcopy(statistics),
        raw_attribute_initialization_seed=initialization_seed,
        raw_attribute_statistics_are_fixed_global_input_statistics_not_source_lookup=True,
        whole_local_raw_bypass_parameters_separate=True)
    model.verify_frozen()
    return dict(original_parameter_objects_preserved=len(named_before),old_attribute_parameter_bindings=old_names,
        old_attribute_Adam_states_preserved=len(slots),new_parameter_names=new_names,
        new_Adam_state_entries=0,new_Adam_starts_at_first_real_gradient=True,
        deliberate_registered_learning_rate_changes=changed,initialization_seed=initialization_seed,
        model_RNG_preservation_requires_caller_verification=True,scientific_optimizer_updates=0)

def resume_raw_optimizers(model,recovery):
    """Restore saved raw architecture by complete verified names, not list positions alone."""
    assert model.config["raw_attribute_stage"]==VERSION
    parameters=dict(model.named_parameters());opts={};bindings=deepcopy(recovery["independent_optimizer_named_bindings"])
    owned=[]
    for branch in ("H","V","attributes"):
        names=bindings[branch];saved=recovery["independent_optimizer_states"][branch]
        expected={n for n,p in parameters.items() if p.requires_grad and (
            n.startswith("events."+branch+".") if branch in ("H","V") else n.startswith(("attributes.","path_attributes.")))}
        assert len(names)==len(set(names)) and set(names)==expected
        groups=[];mapping=[];offset=0
        for group in saved["param_groups"]:
            ids=group["params"];chosen=names[offset:offset+len(ids)];offset+=len(ids)
            assert len(chosen)==len(ids)
            groups.append(dict({k:deepcopy(v) for k,v in group.items() if k!="params"},params=[parameters[n] for n in chosen]))
            mapping.extend(zip(chosen,ids))
        assert offset==len(names)
        opt=torch.optim.AdamW(groups)
        for name,identity in mapping:
            module,relative=name.split(".",1);p=parameters[name]
            assert torch.equal(p.detach().cpu(),recovery["bundle"]["incremental_state"][module][relative].cpu()),name
            if identity in saved["state"]:
                opt.state[p]=_copy_state(saved["state"][identity],p)
                assert _state_equal(opt.state[p],saved["state"][identity]),name
        assert len(opt.state)==len(saved["state"])
        opts[branch]=opt;owned.extend(id(parameters[n]) for n in names)
    assert len(owned)==len(set(owned))
    return opts,bindings
