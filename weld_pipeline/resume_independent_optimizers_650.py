"""TRAIN-only named state migration from final448, including every Adam moment."""
from copy import deepcopy
import torch
from weld_pipeline.physical_trial_transactions_585 import make_independent_optimizers
from weld_pipeline.legacy_incremental_emission_training_loop_457 import state_digest

def resume_optimizers(model,recovery):
    optimizers,bindings=make_independent_optimizers(model)
    assert bindings==recovery["independent_optimizer_named_bindings"]
    for key,opt in optimizers.items():opt.load_state_dict(recovery["independent_optimizer_states"][key])
    old=optimizers["attributes"]
    named=[("attributes."+n,p) for n,p in model.attributes.named_parameters()]
    named += [("path_attributes."+n,p) for n,p in model.path_attributes.named_parameters()]
    rows={"state_boolean":[],"association":[],"numeric_normal":[]}
    for name,parameter in named:
        part=name.split(".")[1]
        key="state_boolean" if part in ("states","binary") else "association" if part in ("association","no_support") else "numeric_normal"
        assert part in ("states","binary","association","no_support","numeric","normals")
        rows[key].append((name,parameter))
    groups=[];names=[]
    for key,rate in (("state_boolean",3e-3),("association",1e-3),("numeric_normal",3e-4)):
        groups.append(dict(params=[p for n,p in rows[key]],lr=rate,phase_name="attributes_"+key))
        names += [n for n,p in rows[key]]
    new=torch.optim.AdamW(groups,weight_decay=1e-4)
    for name,p in named:
        if p in old.state:
            new.state[p]=deepcopy(old.state[p])
            assert state_digest(new.state[p])==state_digest(old.state[p]),name
    assert len(new.state)==len(old.state)
    optimizers["attributes"]=new;bindings["attributes"]=names
    return optimizers,bindings,dict(all_saved_states_preserved=True,attribute_state_count=len(new.state),
        attribute_parameter_names_by_group={key:[n for n,p in value] for key,value in rows.items()},
        learning_rates={g["phase_name"]:g["lr"] for g in groups},optimizer_state_reinitialized=False)
