"""152-channel input adapter and source-frame connection features."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest
from weld_pipeline.sjg_joint_forest_train_1109 import load_case,predict_nodes
from weld_pipeline.sjg_joint_link_runtime_1107 import node_geometry
from weld_pipeline.sjg_interval_forest_runtime_1129 import link_blocks
from weld_pipeline.sjg_relative_geometry_1135 import frame


def load_relative_case(key,feature_run='run329',supervision=False):
    case=load_case(key,supervision,build_features=False)
    f=OUT/feature_run/f'source_{key}';receipt=read(f/'receipt.json')
    assert receipt['columns']==152 and digest(f/'features.npy')==receipt['features_sha256']
    assert receipt['alias_sha256']==case['alias_map_sha256'] and receipt['geometry_sha256']==case['geometry_sha256']
    del case['x'];case['x']=np.load(f/'features.npy',mmap_mode='r')
    assert case['x'].shape==(2*len(case['canonical']),152)
    if supervision:
        with np.load(f/'original_supervision.npz') as z:
            assert np.array_equal(z['labels'],case['y']) and np.array_equal(z['ratios'],case['q'])
            assert np.array_equal(z['canonical'],case['canonical'])
    case['feature_receipt_sha256']=digest(f/'receipt.json');case['relative_feature_run']=feature_run
    geometry_run='run293' if key=='64' else 'run285'
    b=next(r for r in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if r['source']==key)
    surfaces=OUT/f'run185/source_{key}/surfaces.json'
    assert digest(surfaces)==b['source_files']['surfaces.json']
    case['surfaces']=read(surfaces);case['frame_cache']={}
    return case


def source_frame(node,case):
    graph=case['graph'];pid=graph.profile(node['fragment']);key=(pid,node['side'])
    if key not in case['frame_cache']:
        curve=graph.curve(pid);origin=curve.at(np.array([0.]))[0];t0=curve.tangent(np.array([0.]))[0]
        normals=np.zeros((2,3))
        for side,face in enumerate(graph.profiles[pid].get('face_indices',[])[:2]):
            n=np.array(case['surfaces'][face].get('normal',[0,0,0]),float);length=np.linalg.norm(n)
            if length:normals[side]=n/length
        radial=None if curve.kind=='line' else (origin-curve.arc.center_mm)/curve.arc.radius_mm
        ordered=normals if node['side']==0 else normals[::-1]
        E,source=frame(origin,t0,ordered,radial)
        case['frame_cache'][key]=(E,source)
    return case['frame_cache'][key]


def relative_link_blocks(nodes,case,gap_resolver=None):
    """Keep original legal pair set and 32-field schema; rotate only six directions."""
    geometry=[node_geometry(n,case['graph']) for n in nodes]
    tangents=np.array([g[1] for g in geometry]).reshape(-1,2,3)
    for links,features,legal in link_blocks(nodes,case['graph'],gap_resolver=gap_resolver):
        i=int(links[0,0]);E,source=source_frame(nodes[i],case)
        # Preserve the scalar legacy dot-product evaluation order, including tiny near-zero values.
        features[:,8]=np.fromiter((float(tangents[i,1]@tangents[int(j),0]) for j in links[:,1]),np.float64,count=len(links))
        if not source:
            # Zero direction is explicitly unavailable; original node input includes frame_valid=0.
            features[:,2:8]=0
        else:
            a=np.broadcast_to(tangents[i,1]@E,(len(links),3)).copy()
            b=tangents[links[:,1],0]@E
            a[np.abs(a)<=256*np.finfo(float).eps]=0
            b[np.abs(b)<=256*np.finfo(float).eps]=0
            features[:,2:5]=a;features[:,5:8]=b
        assert features.dtype==np.float32 and np.isfinite(features).all(), 'Connection features must be finite float32'
        yield links,features,legal
