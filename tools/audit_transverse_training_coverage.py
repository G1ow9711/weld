"""TRAIN input descriptor availability, not physical plate-height recovery."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 root=BASE/'transverse_endpoint_training_context_v1';assert read(root/'receipt.json')['status']=='EIGHT_TRAIN_TRANSVERSE_ENDPOINT_INPUT_CONTEXT_READY';rows=[]
 for key in read(root/'protocol.json')['keys']:
  r=read(root/key/'receipt.json');assert digest(root/key/'features.npy')==r['features_sha256'];x=np.load(root/key/'features.npy');l=BASE/'predicted_endpoint_attribute_labels_v1'/key;lr=read(l/'receipt.json');assert lr['feature_receipt_sha256']==r['source_feature_receipt_sha256'] and digest(l/'labels.npz')==lr['labels_sha256'];z=np.load(l/'labels.npz');m=z['state_mask'];sides=[]
  for side in [0,1]:
   known=m[:,2+side];has_face=x[:,side*43]>0;boundary=x[:,side*43+4]>0;nonzero_crossing=(x[:,side*43+35]>0)|(x[:,side*43+39]>0);sides.append(dict(side=side,known_required_height=int(known.sum()),with_profile_face=int((known&has_face).sum()),with_searchable_boundary=int((known&boundary).sum()),with_nonzero_crossing=int((known&nonzero_crossing).sum())))
  rows.append(dict(key=key,sides=sides,context_receipt_sha256=digest(root/key/'receipt.json'),label_receipt_sha256=digest(l/'receipt.json')))
 write(BASE/'transverse_training_coverage_audit.json',dict(status='COMPLETE',cases=rows,scope='Availability at frozen predicted endpoints on already geometry-matched TRAIN attribute rows; a nonzero crossing is not validated plate height, missing crossing is not zero height',fit_calls=0,heldout_reads=0,DEV_reads=0))
if __name__=='__main__':main()
