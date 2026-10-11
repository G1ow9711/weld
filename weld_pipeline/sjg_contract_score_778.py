"""Frozen pilot outputs under the existing whole-parent D3/D4 scorer."""
import argparse
import numpy as np
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.contract_geometry_d3_409 import score_source
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_child_evaluation import aggregate

def adapt(predictions):
    parents=[]
    for p in predictions:
        children=[dict(line_segment(s["points"][0],s["points"][-1]),weld_type=p["type"]) for s in p["segments"]]
        gaps=["SHARED" if np.linalg.norm(np.asarray(a["points"][-1])-b["points"][0])<1e-8 else "POSITIVE_GAP"
              for a,b in zip(children[:-1],children[1:])]
        parents.append(dict(child_count=len(children),segments=children,gaps=gaps,is_closed=False))
    return dict(parents=parents,invalid_predictions=[],total_prediction_children=sum(p["child_count"] for p in parents))

def total(scores):
    result={}
    for score in scores:
        for name,c in score["categories"].items():
            if name not in result: result[name]={k:0 for k in ("G","predicted","TP","FP")}
            for k in ("G","predicted","TP","FP"): result[name][k]+=c[k]
    for c in result.values():
        c.update(valid_rate=c["TP"]/c["G"] if c["G"] else None,
                 false_rate_per_G=c["FP"]/c["G"] if c["G"] else None,
                 passed=bool(c["G"] and c["TP"]/c["G"]>.95 and c["FP"]/c["G"]<=.05))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);args=ap.parse_args()
    run=OUT/args.run;receipt=read(run/"receipt.json");inventory=read(OUT/"inventory.json")
    out=run/"contract_whole_parent";out.mkdir(exist_ok=True);result={}
    for arm in receipt["arms"]:
        freezes=read(run/arm/"prediction_freeze.json")
        scores={}
        for item in freezes["outputs"]:
            si=item["source"];path=run/arm/item["path"]
            assert digest(path)==item["sha256"]
            target_path=ROOT/inventory[si]["target"]["target_path"]
            assert digest(target_path)==inventory[si]["target"]["target_sha256"]
            score=score_source(adapt(read(path)),read(target_path));scores[si]=score
            write(out/(arm+"_source_%02d.json"%si),score)
            print("CONTRACT_D3_D4",args.run,arm,si,score["categories"]["overall"],flush=True)
        missing_arc=score_source(adapt([]),read(ROOT/inventory[ARC[0]]["target"]["target_path"]))
        geometry=aggregate({i:next(s for s in receipt["arms"][arm]["scores"] if s["source"]==i) for i in TRAIN})
        result[arm]=dict(train=total([scores[i] for i in TRAIN]),
            development_holdout=total([scores[i] for i in VALID]),
            full8_including_unimplemented_arc=total(list(scores.values())+[missing_arc]),
            child_only_geometry_diagnostic=geometry,
            missing_arc_G=inventory[ARC[0]]["G"],
            E16_status="NOT_IMPLEMENTED_AND_PARTLY_MISSING_REVIEWED_TRUTH",
            pure_XYZ_status="NOT_IMPLEMENTED_CONDITIONAL_STEP_GRAPH_ONLY",
            large_scale_allowed=False,full_contract_pass=False)
    write(out/"summary.json",dict(run=args.run,arms=result,full_contract_pass=False,
        geometry_evaluator_sha256=digest(ROOT/"weld_pipeline/contract_geometry_d3_409.py"),
        predictions_unchanged=True))
if __name__=="__main__":main()
