"""Pure 152/153-dimensional interval-instance runtime with bounded profile loading."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest
from weld_pipeline.sjg_seven_relative_prepare_1146 import feature_path
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_interval_forest_runtime_1129 import predict_instances
from weld_pipeline.sjg_relative_forest_runtime_1139 import relative_link_blocks


def load_case(key):
    folder=feature_path(key);receipt=read(folder/'receipt.json')
    assert receipt['columns']==152 and digest(folder/'features.npy')==receipt['features_sha256']
    alias_run='run293' if key=='64' else 'run288';geometry_run='run293' if key=='64' else 'run285'
    alias=OUT/alias_run/f'source_{key}_alias_map.npz';assert digest(alias)==receipt['alias_sha256']
    with np.load(alias) as z:canonical=z['canonical_fragments']
    binding=next(v for v in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
    assert binding['geometry_sha256']==receipt['geometry_sha256']
    graphfile=OUT/geometry_run/binding['geometry_file'];assert digest(graphfile)==binding['geometry_sha256']
    profilefile=OUT/f'run185/source_{key}/profiles.json';surfacefile=profilefile.parent/'surfaces.json'
    assert digest(profilefile)==binding['source_files']['profiles.json'] and digest(surfacefile)==binding['source_files']['surfaces.json']
    profiles=read_profile_geometry(profilefile,retain_fields=('support_intervals_mm','unobserved_gaps_mm'))
    with np.load(graphfile) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
    x=np.load(folder/'features.npy',mmap_mode='r');assert x.shape==(2*len(canonical),152)
    return dict(key=key,x=x,canonical=canonical,graph=graph,surfaces=read(surfacefile),frame_cache={},
        feature_receipt_sha256=digest(folder/'receipt.json'),alias_map_sha256=receipt['alias_sha256'],geometry_sha256=receipt['geometry_sha256'])
