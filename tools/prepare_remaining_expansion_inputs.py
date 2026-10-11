"""Prepare XYZ-only inputs for the first registered structure-coverage cohort."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'server_rebuild/expansion_v1';DATA=ROOT.parent/'data/G_Data'
def main():
 groups={g['key']:g for g in read(B/'group_manifest.json')['groups']};summary=read(B/'training_label_rebuild_summary.json');chosen=[]
 for receipt in summary['successes']:
  key=receipt['source_key'];t=read(B/'labels'/key/'targets.json');v3=sum(len(p['children'])==3 and all(c['weld_type']=='vertical' for c in p['children']) for p in t['parents']);arcs=sum(c['original_kind']=='arc' for p in t['parents'] for c in p['children'])
  if not (B/'inputs'/key/'receipt.json').exists():chosen.append(dict(key=key,family_id=groups[key]['family_id'],directory=receipt['directory'],vertical_three_child_parents=v3,arc_children=arcs,labels_receipt_sha256=digest(B/'labels'/key/'receipt.json')))
 protocol=B/'additional_train_input_protocol.json';assert not protocol.exists();write(protocol,dict(cases=chosen,selection_rule='All remaining accepted expansion TRAIN families without existing XYZ input receipts; no model scores or validation labels used',group_manifest_sha256=digest(B/'group_manifest.json'),same_candidate_budget_training_comparison_required=True,old_weights_frozen=True,DEV64_excluded=True,expansion_validation_labels_unused=True,formal_acceptance=False,source_sha256=digest(__file__)))
 for case in chosen:
  key=case['key'];assert groups[key]['role']=='EXPANSION_TRAIN';folder=DATA/case['directory'];files=list(folder.glob('*weld_workpiece.ply'));assert len(files)==1;path=files[0]
  with path.open() as f:
   header=[]
   for line in f:
    header.append(line.strip())
    if line.strip()=='end_header':break
   assert 'format ascii 1.0' in header and 'comment unit millimeter; frame reference_workpiece' in header;n=int(next(s.split()[-1] for s in header if s.startswith('element vertex')));xyz=np.loadtxt(f,usecols=(0,1,2),dtype=np.float64,ndmin=2);assert xyz.shape==(n,3) and np.isfinite(xyz).all()
  import shutil
  assert shutil.disk_usage(B).free>5*1024**3+xyz.nbytes,'Preserve 5GiB disk headroom'
  dest=B/'inputs'/key;dest.mkdir(parents=True,exist_ok=False);np.save(dest/'xyz_mm.npy',xyz);write(dest/'receipt.json',dict(coordinate_frame='workpiece_local',unit='mm',original_points=n,coordinate_transform_applied=False,scale_applied=1.,only_XYZ_used=True,header=header,source_file=str(path.relative_to(DATA)),source_sha256=digest(path),xyz_sha256=digest(dest/'xyz_mm.npy'),bbox_mm=[xyz.min(0).tolist(),xyz.max(0).tolist()],role='EXPANSION_TRAIN',family_id=case['family_id'],selection_protocol_sha256=digest(protocol),input_extraction_GT_reads=0,STEP_reads=0));print(key,n,flush=True)
if __name__=='__main__':main()
