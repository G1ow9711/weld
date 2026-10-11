"""Freeze all pure-input parent graphs before reading TRAIN parent labels."""
import argparse
import gc
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_parent_inputs_829 import interval_inputs,continuation_inputs,training_targets

BASE=OUT/'run45'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);inventory=read(OUT/'inventory.json')
    write(dest/'registered_protocol.json',dict(stage='PURE_POINT_INTERVAL_AND_PARENT_GRAPH_DATA',
        source_sha256=digest(__file__),features_sha256=digest(ROOT/'weld_pipeline/sjg_parent_inputs_829.py'),
        sources=TRAIN+VALID+ARC,train=TRAIN,original_G=496,base_prediction_run='run45',
        node_features=132,edge_features=20,both_traversal_orientations=True,
        material_ID_not_used=True,STEP_reads=0,all_inputs_frozen_before_parent_labels=True,
        known_parent_ID_only_training_target=True,curves_pending=True,full_contract_pass=False,start_utc=utc()))
    try:
        assert read(BASE/'receipt.json')['status']=='FREE_SEQUENCE_STRICT_GEOMETRY_EVALUATED'
        frozen=[]
        for si in TRAIN+VALID+ARC:
            check_resources();children,profiles,x=interval_inputs(si,BASE);pairs,relation=continuation_inputs(children,profiles)
            path=dest/f'source_{si:02d}_inputs.npz'
            np.savez_compressed(path,nodes=x,pairs=pairs,relation=relation)
            frozen.append(dict(source=si,file=path.name,sha256=digest(path),nodes=len(x),edges=len(pairs),
                prediction_sha256=digest(BASE/f'source_{si:02d}_predictions.json')))
            print('PURE_PARENT_INPUT_FROZEN',si,'CHILDREN',len(x),'ORIENTED_EDGES',len(pairs),flush=True)
            del children,profiles,x,pairs,relation;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=frozen,all_before_GT=True))
        rows=[]
        for si in TRAIN:
            children=read(BASE/f'source_{si:02d}_predictions.json')['predictions']
            with np.load(dest/f'source_{si:02d}_inputs.npz') as z:pairs=z['pairs']
            target_path=ROOT/inventory[si]['target']['target_path'];assert digest(target_path)==inventory[si]['target']['target_sha256']
            target=read(target_path);node,edge,matches=training_targets(children,pairs,target)
            np.savez_compressed(dest/f'source_{si:02d}_targets.npz',node=node,edge=edge)
            write(dest/f'source_{si:02d}_target_bindings.json',dict(matches=matches,GT_ID_not_input=True,original_G=target['original_children']))
            row=dict(source=si,original_G=target['original_children'],nodes=len(node),positive_nodes=int(node.sum()),
                edges=len(edge),positive_links=int(edge[:,0].sum()),positive_shared=int(edge[:,1].sum()))
            rows.append(row);print('PURE_PARENT_TRAIN_TARGETS',row,flush=True)
        write(dest/'receipt.json',dict(status='PURE_PARENT_INPUTS_AND_TRAIN_TARGETS_COMPLETE',rows=rows,
            end_utc=utc(),optimizer_updates=0,full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
