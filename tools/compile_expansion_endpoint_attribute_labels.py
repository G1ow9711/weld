"""Compile explicitly masked process states; no model input or geometry change."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';SRC=BASE/'excel_endpoint_sidecars_v2';DEST=BASE/'endpoint_attribute_supervision_v1'
FIELDS=[('start','blocker_distance','front_wall'),('end','blocker_distance','back_wall'),('start','plate_height','start_height'),('end','plate_height','end_height'),('start','hole_height','start_hole_height'),('end','hole_height','end_hole_height'),('start','outer_edge_distance','start_base_edge'),('end','outer_edge_distance','end_base_edge')]
ENCODING={'EXACT_MM':0,'DISTANCE_GT_50':1,'NO_HOLE':2,'NO_BLOCKER_OR_DISTANCE_GT_50':3}
def main():
 DEST.mkdir(exist_ok=False);write(DEST/'protocol.json',dict(fields=[f[2] for f in FIELDS],state_codes=ENCODING,unknown_state=-1,non_exact_states_have_nan_values=True,optional_exact_values_retained_but_masked=True,unknown_not_zero_or_negative=True,vertical_nonrequired_fields_masked=True,no_parent_flags_broadcast=True,no_plate_normals_invented=True,no_model_fit=True,source_protocol_sha256=digest(SRC/'protocol.json')));reports=[]
 for entry in read(SRC/'summary.json')['cases']:
  key=entry['key'];path=SRC/key/'evidence.json';assert digest(path)==entry['evidence_sha256'];r=read(path);records=[];state=[];values=[];required=[]
  for parent in r['parents']:
   for child in parent['children']:
    st=[];v=[];req=[]
    for side,field,name in FIELDS:
     label=child['endpoint_attributes'][side][field];code=ENCODING[label['encoding']] if label['known'] else -1;st.append(code);v.append(float(label['raw_value']) if code==0 else np.nan);req.append(label['required_by_current_customer_rule'])
    records.append(dict(parent_id=parent['parent_id'],child_id=child['child_id'],child_index=child['index'],weld_type=child['weld_type']));state.append(st);values.append(v);required.append(req)
  state=np.asarray(state,np.int8);values=np.asarray(values,np.float64);required=np.asarray(required,bool);sm=required&(state>=0);nm=sm&(state==0);assert np.isfinite(values[nm]).all();assert np.isnan(values[state!=0]).all()
  dest=DEST/key;dest.mkdir();np.savez(dest/'labels.npz',state=state,exact_mm=values,required_mask=required,state_mask=sm,numeric_mask=nm);write(dest/'records.json',records);report=dict(key=key,children=len(records),known_required_states=int(sm.sum()),required_exact_values=int(nm.sum()),required_unknown=int((required&~sm).sum()),state_histogram={str(c):int((sm&(state==c)).sum()) for c in range(4)},source_evidence_sha256=digest(path),labels_sha256=digest(dest/'labels.npz'),records_sha256=digest(dest/'records.json'),model_fit_calls=0,DEV_reads=0);write(dest/'receipt.json',report);reports.append(report);print(key,report['known_required_states'],flush=True)
 write(DEST/'summary.json',dict(cases=reports,children=sum(r['children'] for r in reports),known_required_states=sum(r['known_required_states'] for r in reports),required_exact_values=sum(r['required_exact_values'] for r in reports),required_unknown=sum(r['required_unknown'] for r in reports),model_fit_calls=0,DEV_reads=0))
if __name__=='__main__':main()
