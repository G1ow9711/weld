"""Post-round analysis. Target-field decoding is an upper-bound control only."""
import json
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_geometry_776 import read,write
from weld_pipeline.sjg_train_776 import OUT,TRAIN,VALID,decode_free,score_frozen
from weld_pipeline.contract_child_evaluation import aggregate

def main():
    run=OUT/"run02";receipt=read(run/"receipt.json");inventory=read(OUT/"inventory.json")
    out=OUT/"round01_review";out.mkdir(exist_ok=True)
    summary=dict(status="LOCAL_REVIEW_NO_EXTERNAL_MESSAGE",training_updates=receipt["total_optimizer_updates"],
        scope="CONDITIONAL_STRUCTURE_CHILD_GEOMETRY_ONLY_NOT_PCD_ACCEPTANCE",arms={},
        full_contract_pass=False,large_scale_allowed=False)
    for arm in ("local_classifier","relation_graph"):
        scores={s["source"]:s for s in receipt["arms"][arm]["scores"]}
        train=aggregate({i:scores[i] for i in TRAIN});holdout=aggregate({i:scores[i] for i in VALID})
        for values,ids in ((train,TRAIN),(holdout,VALID)):
            matches=[m for i in ids for m in scores[i]["matches"]]
            values["matched_endpoint_max_mm"]=max((max(m["start_mm"],m["end_mm"]) for m in matches),default=None)
            values["matched_whole_line_max_mm"]=max((m["whole_line"]["lower_mm"] for m in matches),default=None)
            values["geometry_errors_are_for_matched_children_only"]=True
        summary["arms"][arm]=dict(train=train,development_holdout=holdout)
        print(arm,json.dumps({role:{k:v[k] for k in ["truth_count","valid_children","false_predictions","valid_extraction_rate","false_rate_per_truth"]}
              for role,v in summary["arms"][arm].items()}),flush=True)
    write(out/"summary.json",summary)
    controls={}
    for si in TRAIN:
        folder=OUT/f"source_{si:02d}"
        with np.load(folder/"inputs.npz",allow_pickle=False) as z:
            arrays={k:z[k] for k in ("ptr",)}
        arrays["contacts"]=read(folder/"geometry.json")["contacts"]
        with np.load(folder/"supervision_v2.npz",allow_pickle=False) as z:
            pred=decode_free(arrays,z["targets"],z["offsets"])
        score=score_frozen(pred,si,inventory)
        score["scope"]="GT_TARGET_FIELD_DECODER_CONTROL_NOT_MODEL_ACCURACY"
        write(out/f"target_decoder_control_{si:02d}.json",score);controls[si]=score
        print("TARGET_FIELD_CONTROL",si,score["valid_children"],"/",score["truth_count"],
              "FP",score["false_predictions"],flush=True)
    write(out/"target_decoder_control_total.json",dict(scope="GT_FIELDS_ONLY_NOT_MODEL",**aggregate(controls)))
if __name__=="__main__":main()
