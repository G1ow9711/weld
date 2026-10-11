"""Execute route copies on unchanged original TRAIN21 and compare actual outputs."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'server_rebuild/expansion_v1/routing_controls';KEY='21'
def main():
 DEST.mkdir(exist_ok=False);rows=[]
 import tools.prepare_expansion_child_features as child
 child.BASE=ROOT/'server_rebuild/best_baseline_restore';child.DEST=DEST/'child';child.run(KEY)
 def compare(actual,expected,names):
  for name in names:
   a=actual/name;b=expected/name
   if name.endswith('.npy'):assert np.array_equal(np.load(a),np.load(b)),name
   else:assert read(a)==read(b),name
   rows.append(dict(output=str(a.relative_to(ROOT)),reference=str(b.relative_to(ROOT)),actual_sha256=digest(a),reference_sha256=digest(b),equal=True))
 compare(child.DEST/('source_'+KEY),ROOT/'server_rebuild/child_residual_v1'/('source_'+KEY),['features.npy','origins.json','metadata.json','parents.json'])
 import tools.generate_expansion_parent_hypotheses as hypotheses
 hypotheses.BASE=ROOT/'server_rebuild/best_baseline_restore';hypotheses.INPUT=child.DEST;hypotheses.DEST=DEST/'hypotheses';hypotheses.run(KEY)
 compare(hypotheses.DEST/('source_'+KEY),ROOT/'server_rebuild/parent_hypotheses_v1'/('source_'+KEY),['hypotheses.json','atoms.json','paths.json','truncation.json','edges.json'])
 import tools.prepare_expansion_parent_scoring_inputs as unified
 fake=DEST/'project_root';exp=fake/'server_rebuild/expansion_v1';exp.mkdir(parents=True)
 (exp/'frozen_features').symlink_to(ROOT/'server_rebuild/best_baseline_restore',target_is_directory=True)
 (exp/'parent_hypotheses_v1').symlink_to(hypotheses.DEST,target_is_directory=True)
 unified.ROOT=fake;unified.DEST=DEST/'unified';unified.run(KEY)
 compare(unified.DEST/('source_'+KEY),ROOT/'server_rebuild/unified_parent_scoring_v1'/('source_'+KEY),['features.npy','child_features.npy','child_records.json','child_metadata.json','child_segments.json','parent_children.json','parents.json','default_configuration.json'])
 write(DEST/'receipt.json',dict(status='PASS',scope='Actual original TRAIN21 outputs; proves path copies retain this case numerics and structures, not expansion data quality or full-cloud invariance',key=KEY,rows=rows,GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,routing_source_manifest_sha256=digest(DEST.parent/'parent_feature_routing_sources.json')))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
