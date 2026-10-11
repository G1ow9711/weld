"""Prepare the registered DEV64 pure inputs without reading DEV targets."""
import gc
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_alias_normalization_1108 import normalize
from weld_pipeline.sjg_joint_object_features_1106 import prepare


def main():
    key='64';run='run293';dest=OUT/run;dest.mkdir(exist_ok=False);start=time.perf_counter();check_resources()
    metadata=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    base=OUT/f'run185/source_{key}';binding=read(base/'input_freeze.json')
    for name in ('profiles.json','surfaces.json','surface_support.npz'):
        assert digest(base/name)==binding['files'][name]
    assert digest(OUT/metadata['input_file'])==metadata['input_sha256']
    profiles=read(base/'profiles.json');graph=JointSegmentGraph.build(profiles)
    geometry_path=dest/f'source_{key}_geometry.npz'
    np.savez(geometry_path,offsets=graph.offsets,spans=graph.spans)
    row=dict(source=key,profiles=len(profiles),fragments=len(graph.spans),source_files=binding['files'],
             geometry_file=geometry_path.name,geometry_sha256=digest(geometry_path),target_reads=0)
    write(dest/'all_geometry_frozen_before_labels.json',dict(cases=[row],target_reads=0,
          code_sha256=digest(ROOT/'weld_pipeline/sjg_joint_segment_graph_1104.py'),utc=utc()))
    rep,prep,merges=normalize(profiles,graph.offsets,graph.spans)
    alias_path=dest/f'source_{key}_alias_map.npz'
    np.savez(alias_path,fragment_representative=rep,profile_representative=prep,canonical_fragments=np.unique(rep))
    write(dest/f'source_{key}_proven_aliases.json',dict(merges=merges,
          surface_support_sha256=binding['files']['surface_support.npz'],target_reads=0))
    normalized=dict(source=key,original_fragments=len(rep),canonical_fragments=len(np.unique(rep)),
                    map_sha256=digest(alias_path),labels_read=0)
    write(dest/'all_alias_maps_frozen_before_labels.json',dict(cases=[normalized],labels_read=0,
          code_sha256=digest(ROOT/'weld_pipeline/sjg_joint_alias_normalization_1108.py'),utc=utc()))
    del profiles,graph,rep,prep;gc.collect()
    prepare(key,run,geometry_run=run)
    write(dest/'receipt.json',dict(status='PURE_DEV_INPUT_READY_UNSCORED',source=key,
        original_cohort_role=metadata['role'],input_sha256=metadata['input_sha256'],
        geometry=row,normalized=normalized,object_feature_receipt_sha256=digest(dest/f'source_{key}/receipt.json'),
        target_reads=0,STEP_reads=0,forest_fit_calls=0,model_prediction_runs=0,
        extraction_correct_rate=None,extraction_precision_mm=None,seconds=time.perf_counter()-start,utc=utc()))
    print('DEV64_INPUT_READY_NO_LABELS_NO_MODEL_TEST',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
