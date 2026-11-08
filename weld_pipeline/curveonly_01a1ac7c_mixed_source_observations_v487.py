"""Actual raw XYZ and finite-material observations for new bounded sources.

The existing raw34 calculation is reused verbatim. All finite12, solid74 and
direction23 observations are recomputed on the new geometry and real materials.
Original candidate geometry and feature rows are retained exactly.
"""
import copy,inspect,time
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_frame_v30 as frame
from . import curveonly_01a1ac7c_exact_mesh_index_v110 as finite
from . import curveonly_01a1ac7c_material_sections_v124 as volume
from . import curveonly_01a1ac7c_section_local_v135 as section
from . import curveonly_01a1ac7c_material_direction_v160 as direction
from . import curveonly_01a1ac7c_direction_spans_v247 as encoder


def raw34(candidates,xyz,base,pose):
    source=inspect.getsource(cad.input_curves)
    marker='    begin=time.perf_counter();tree=cKDTree(xyz,balanced_tree=False);'
    assert source.count(marker)==1
    body=source[source.index(marker):]
    text='def supplied(candidates,xyz,base,audit):\n    start=time.perf_counter();timing={}\n'+body
    ns=dict(cad.__dict__);exec(compile(text,__file__,'exec'),ns)
    values,packet=ns['supplied'](copy.deepcopy(candidates),xyz,np.asarray(base),copy.deepcopy(pose))
    return frame.canonical(values,packet)


def measured_other_distances(packet,solids):
    transform=np.asarray(packet['pose']['transform_STEP_to_XYZ']);R,t=transform[:3,:3],transform[:3,3]
    faces=[list(cad.items(s,cad.TopAbs_FACE)) for s in solids]
    boxes=np.asarray([cad.bbox(s) for s in solids]);fboxes=[np.asarray([cad.bbox(f) for f in fs]) for fs in faces]
    owner_at=finite.owner_function(packet)
    for ci,c in enumerate(packet['candidates']):
        points=np.asarray(c['points']);s=cad.arclength(points);ids=[0,len(points)//2,len(points)-1];other=[]
        for index in ids:
            q=(points[index]-t)@R;owner=owner_at(ci,float(s[index]),packet);distance=1e4
            for mi,fs in enumerate(faces):
                if mi==owner:continue
                box=boxes[mi];lower=np.linalg.norm(np.maximum(np.maximum(box[:3]-q,q-box[3:]),0))
                if lower>min(distance,20.):continue
                fb=fboxes[mi];bounds=np.linalg.norm(np.maximum(np.maximum(fb[:,:3]-q,q-fb[:,3:]),0),axis=1)
                for fi in np.argsort(bounds):
                    if bounds[fi]>min(distance,20.):break
                    distance=min(distance,cad.finite_distance(fs[int(fi)],q))
            other.append(float(distance))
        c['other_distance']=other


def append(features,packet,generated,xyz,solids):
    start=time.perf_counter();count=len(packet['candidates']);features=np.asarray(features)
    wanted=sorted({int(p['curve']) for c in generated['candidates'] for p in c['source_material_path']})
    if not wanted:return features,packet,dict(new_candidates=0,original_prefix_exact=True)
    mapping={old:i for i,old in enumerate(wanted)};candidates=[]
    for old in wanted:
        c=copy.deepcopy(generated['primitives'][old]);c.update(source_curve_count=1,
            original_native_members=[dict(curve=mapping[old],reversed=False)],
            input_geometry_connections=[],joint_identity_accepted=False)
        candidates.append(c)
    for value in generated['candidates']:
        c=copy.deepcopy(value)
        c['source_material_path']=[dict(p,curve=mapping[p['curve']]) for p in c['source_material_path']]
        c['composite_carrier_members']=copy.deepcopy(c['source_material_path'])
        c['material_extent']=candidates[c['source_material_path'][0]['curve']]['material_extent']
        candidates.append(c)
    local=dict(candidates=candidates,pose=copy.deepcopy(packet['pose']))
    measured_other_distances(local,solids)
    raw,local=raw34(candidates,xyz,packet['base_axis'],packet['pose'])
    # Independently replay the pre-existing raw calculation before reuse.
    replay,old=raw34(packet['candidates'],xyz,packet['base_axis'],packet['pose'])
    replay_error=float(np.max(np.abs(replay-features[:,:34])))
    assert packet['offsets']==old['offsets'] and replay_error<=1e-5,replay_error
    fv,fr=finite.build(local,solids)
    local['local_finite_material_v94']=dict(values=fv.tolist(),receipt=fr)
    sv,sr=volume.build(local,solids)
    new_features,local=section.augment(raw,local,sv,sr)
    dv,dr=direction.build(local,solids)
    local['local_material_direction_v160']=dict(values=dv.tolist(),receipt=dr)
    rows=encoder.candidate_rows(new_features,local)
    assert all(row['grid'].shape[1]==134 for row in rows)
    combined=copy.deepcopy(packet);added=copy.deepcopy(local['candidates'])
    for c in added:
        for key in ['source_material_path','composite_carrier_members','original_native_members']:
            for p in c.get(key,[]):p['curve']+=count
        for connection in c.get('source_connections',[]):
            connection['curve']=mapping[connection['curve']]+count
            connection['line']=mapping[connection['line']]+count
    combined['candidates'].extend(added)
    combined['offsets'].extend(packet['offsets'][-1]+x for x in local['offsets'][1:])
    for name in ['local_finite_material_v94','local_material_direction_v160']:
        combined[name]['values'].extend(local[name]['values'])
        combined[name]['receipt']=dict(original=packet[name]['receipt'],extension=local[name]['receipt'],
            original_values_retained_exact=True,new_values_from_actual_finite_material=True)
    output=np.concatenate([features,new_features])
    assert np.array_equal(output[:len(features)],features)
    assert combined['candidates'][:count]==packet['candidates']
    # Resolve each extended path against the combined original-material list.
    owner_at=finite.owner_function(combined)
    for ci,c in enumerate(added,count):
        for p in c.get('source_material_path',[]):
            assert owner_at(ci,(p['start_mm']+p['end_mm'])/2,combined)==p['material']
    receipt=dict(original_candidates=count,new_candidates=len(added),new_stations=len(new_features),
        new_mixed_candidates=[count+i for i,c in enumerate(local['candidates']) if c.get('source_material_path')],
        certified_linear_provenance_only_candidates=[count+i for i,c in enumerate(local['candidates']) if c.get('native_geometry_kind')=='CERTIFIED_NATIVE_LINEAR'],
        original_prefix_exact=True,old_raw34_replay_max_error=replay_error,
        raw34_verbatim_original_calculation=True,real_XYZ_tree_and_native_material_queries=True,
        finite12_solid74_direction23_recomputed=True,no_fabricated_features_or_should_weld_decisions=True,
        seconds=time.perf_counter()-start)
    combined['mixed_source_observations_v487']=receipt
    return output,combined,receipt
