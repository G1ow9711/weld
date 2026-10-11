"""Experiment A prep: freeze a label-free conditional CAD graph, then label it."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import gmsh
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest,finite_faces,construct_contacts,graph_arrays,serial
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776"
TRAIN=[21,4,10,25,36,42]
VALID=[64]
ARC=[24]

def make_targets(contacts,arrays,target):
    ptr=arrays["ptr"]; y=np.zeros((int(ptr[-1]),3),np.float32)
    offsets=np.zeros((len(y),2),np.float32); offset_mask=np.zeros_like(offsets)
    assignments=[];missing=[];ambiguous=[]
    for pi,p in enumerate(target["parents"]):
        if any(c["original_kind"]!="line" for c in p["children"]):
            missing.append(dict(parent=pi,children=p["original_child_count"],reason="NONLINEAR_NOT_EXPRESSED"));continue
        for si,child in enumerate(p["children"]):
            ends=np.asarray(child["world_representation"]["endpoints_mm"])
            choices=[]
            for ci,c in enumerate(contacts):
                q=(ends-c["p0"])@c["direction"]
                projected=c["p0"]+q[...,None]*c["direction"]
                error=float(np.linalg.norm(projected-ends,axis=-1).max())
                if error<=.5 and q.min()>=-.2 and q.max()<=c["length"]+.2 and len(set(c["materials"]))==2:
                    choices.append((error,ci,q))
            if not choices:
                missing.append(dict(parent=pi,child=si,children=1,reason="NO_FINITE_MATERIAL_CONTACT"));continue
            choices.sort(key=lambda z:(z[0],z[1]))
            # Supervise children across distinct physical contacts without changing
            # their original parent. Parent-path recovery is evaluated separately.
            _,ci,qs=choices[0]
            if len(choices)>1: ambiguous.append(dict(parent=pi,child=si,candidates=[c[1] for c in choices]))
            c=contacts[ci];start,stop=ptr[ci:ci+2]
            q=np.linspace(0,c["length"],stop-start)
            qa,qb=qs
            lo,hi=sorted([float(qa),float(qb)])
            y[start:stop,0]=np.maximum(y[start:stop,0],((q>=lo)&(q<=hi)).astype(float))
            for ei,v in enumerate([lo,hi]):
                prob=np.exp(-.5*((q-v)/2.)**2)
                better=prob>y[start:stop,ei+1]
                y[start:stop,ei+1]=np.maximum(y[start:stop,ei+1],prob)
                offsets[start:stop,ei][better]=((v-q)/2)[better]
                offset_mask[start:stop,ei][better]=(np.abs(q-v)<=3)[better]
            assignments.append(dict(parent=pi,child=si,contact=ci,lo=lo,hi=hi,directed=bool(qb>=qa)))
    return dict(targets=y,offsets=offsets,offset_mask=offset_mask),dict(
        original_G=target["original_children"],represented_children=len(assignments),
        misses=missing,ambiguous_material_assignments=ambiguous,assignments=assignments,
        material_pair_customer_truth_available=False,conditional_geometry_only=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=int,required=True);args=ap.parse_args()
    row=read(OUT/"inventory.json")[args.source]
    folder=OUT/("source_%02d"%args.source);folder.mkdir(exist_ok=False)
    started=time.monotonic()
    try:
        frames=row["step"]["verified_offline_frames"]
        if not frames: raise RuntimeError("No offline structure reference; pure XYZ branch required")
        # Frame 0 is preselected from the existing source book; no GT-dependent
        # frame selection. Alternative frames and scope remain in the manifest.
        frame=frames[0];transform=np.asarray(frame["CAD_to_world"])
        pcd=ROOT/row["raw"]["original_pcd_path"]
        step=ROOT/row["step"]["STEP_path"];catalogue=ROOT/frame["catalogue_path"]
        assert digest(pcd)==row["raw"]["original_pcd_sha256"]
        assert digest(step)==row["step"]["STEP_sha256"]
        assert digest(catalogue)==frame["catalogue_sha256"]
        xyz=read_xyz_pcd(pcd)*1000.
        assert len(xyz)==row["points"]
        print("PREP_SOURCE",args.source,row["name"],"PCD_POINTS",len(xyz),"NO_EPOCH_PREPROCESSING",flush=True)
        gmsh.initialize();gmsh.option.setNumber("General.Terminal",0)
        gmsh.model.occ.importShapes(str(step));gmsh.model.occ.synchronize()
        faces=finite_faces(read(catalogue),transform)
        contacts=construct_contacts(faces,transform)
        arrays=graph_arrays(faces,contacts,xyz)
        np.savez_compressed(folder/"inputs.npz",**arrays)
        write(folder/"geometry.json",serial(dict(contacts=contacts,faces=faces)))
        freeze=dict(source=args.source,role="TRAIN" if args.source in TRAIN else "DEVELOPMENT_HOLDOUT",
            pcd_sha256=digest(pcd),step_sha256=digest(step),frame=frame,
            frame_alternatives_count=len(frames),geometry_sha256=digest(folder/"geometry.json"),
            input_sha256=digest(folder/"inputs.npz"),faces=len(faces),contacts=len(contacts),
            queries=len(arrays["query"]),edges=arrays["edge_index"].shape[1],
            generation_weld_GT_reads=0,STEP_is_conditional_experiment_input=True,
            deployable_pure_XYZ=False,formal_independent=False)
        write(folder/"input_freeze.json",freeze)
        gmsh.finalize();del xyz;gc.collect()
        target_path=ROOT/row["target"]["target_path"]
        assert digest(target_path)==row["target"]["target_sha256"]
        labels,coverage=make_targets(contacts,arrays,read(target_path))
        np.savez_compressed(folder/"supervision.npz",**labels)
        write(folder/"coverage.json",coverage)
        write(folder/"ready.json",dict(**freeze,original_G=coverage["original_G"],
            represented_children=coverage["represented_children"],elapsed_seconds=time.monotonic()-started,
            completed=True,optimizer_updates=0))
        print("PREP_DONE",args.source,"contacts",len(contacts),"queries",len(arrays["query"]),
              "REPRESENTED",coverage["represented_children"],"/",coverage["original_G"],flush=True)
    except BaseException as e:
        import traceback
        write(folder/"failure.json",dict(error=str(e),traceback=traceback.format_exc(),optimizer_updates=0))
        raise
if __name__=="__main__":main()
