"""Real candidate controls for reusing the frozen link head in pool expansion."""
from pathlib import Path
import pickle
import numpy as np
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_child_parent_supervision_1255 import singleton,state_geometry,link_features
from weld_pipeline.sjg_child_parent_model_1256 import controls
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.sjg_relative_geometry_1135 import transformations

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore';DEST=ROOT/'server_rebuild/parent_residual_v1'
def main():
 mp=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run446/models.pkl'
 assert digest(mp)==read(mp.parent/'training_receipt.json')['model_sha256']
 with mp.open('rb') as f:model=pickle.load(f)['models']['link']
 controls(DEST);reports=[]
 for key in ['21','BM79A_native_local','10','25','36']:
  folder=BASE/'legacy_runtime'/('source_'+key)/'run379'/('test_'+key)
  allparents=read(folder/'decoded.json')['parents'];metadata=read(folder/'parent_metadata.json');pairs=[]
  for pi in np.unique(np.linspace(0,len(allparents)-1,min(24,len(allparents))).astype(int)):
   for segment in allparents[pi]['segments']:
    pairs.append((singleton(segment),dict(metadata[pi],length_mm=segment['length_mm'],line_fraction=float(segment['kind']=='line'))))
  state=state_geometry(ParentRelations([p for p,m in pairs],[m for p,m in pairs]));n=len(state['ends'])
  edges=np.array([(a,b) for a in range(n) for b in range(n) if a//2!=b//2],int).reshape(-1,2)
  reference=link_features(state,edges);prob=model.predict_proba(reference);rows=[]
  for name,rotation,shift in transformations():
   moved=dict(state,ends=state['ends']@rotation.T+shift,tangent=state['tangent']@rotation.T)
   features=link_features(moved,edges);q=model.predict_proba(features)
   rows.append(dict(transform=name,feature_max_abs=float(abs(features-reference).max()),probability_max_abs=float(abs(q-prob).max()),class_changes=int((q.argmax(1)!=prob.argmax(1)).sum())))
  # Row permutation must not affect the frozen forest's probabilities.
  assert np.array_equal(model.predict_proba(reference[::-1])[::-1],prob)
  reports.append(dict(source=key,children=len(pairs),arc_children=sum(p['segments'][0]['kind']=='arc' for p,m in pairs),directed_pairs=len(edges),transformations=rows,row_permutation_exact=True))
 report=dict(cases=reports,model_sha256=digest(mp),GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,controls_scope='Frozen link features and probabilities only; not whole XYZ pipeline invariance',feature_tolerance=1e-4)
 report['feature_controls_passed']=all(t['feature_max_abs']<1e-4 for r in reports for t in r['transformations'])
 report['forest_probability_invariant_exact']=all(t['probability_max_abs']==0 for r in reports for t in r['transformations'])
 write(DEST/'frozen_link_controls.json',report);print(report,flush=True)
if __name__=='__main__':main()
