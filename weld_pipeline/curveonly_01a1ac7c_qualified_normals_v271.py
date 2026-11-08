"""Qualified finite-face normals with explicit volume/mesh uncertainty."""
import time
import itertools
import numpy as np
from OCP.BRepCheck import BRepCheck_Analyzer
from . import curveonly_01a1ac7c_surface_normals_v269 as base
original,indexed,shadow,reference,direction=base.original,base.indexed,base.shadow,base.reference,base.direction
DEFLECTION_MM,SUPPORT_BAND_MM,TIE_MM,LAYER_TIE_MM=base.DEFLECTION_MM,base.SUPPORT_BAND_MM,base.TIE_MM,base.LAYER_TIE_MM
DIRECTION_EPS_MM=base.DIRECTION_EPS_MM
NAMES=base.NAMES+('owner_volume_valid','nearest_layer1_all_volumes_valid','nearest_layer2_all_volumes_valid','all_material_volumes_valid',
    'nearest_layer1_mesh_distance_available','nearest_layer1_mesh_distance_log_mm','nearest_layer2_mesh_distance_available','nearest_layer2_mesh_distance_log_mm')
assert len(NAMES)==33


def box_lower(points,boxes):
    if not len(boxes):return np.full(len(points),np.inf)
    boxes=np.asarray(boxes,float)
    delta=np.maximum(np.maximum(boxes[:,:3][None]-points[:,None],points[:,None]-boxes[:,3:][None]),0.)
    return np.linalg.norm(delta,axis=2).min(1)


def closest(points,record,frames,cache=None):
    if not len(record['mesh']):
        return dict(distance_mm=np.full(len(points),np.inf),normal_available=np.zeros(len(points),bool),
            normal_min=np.zeros((len(points),3)),normal_max=np.zeros((len(points),3)),support_triangles=np.zeros(len(points),np.int64))
    out=base.closest(points,record['mesh'],frames,cache)
    qualified=out['normal_available'].copy()
    if len(record['bad_mesh']):
        bad=base.closest(points,record['bad_mesh'],frames,cache)['distance_mm']
        qualified &= bad>out['distance_mm']+SUPPORT_BAND_MM+TIE_MM
    qualified &= box_lower(points,record['missing_boxes'])>out['distance_mm']+SUPPORT_BAND_MM+TIE_MM
    out['normal_available']=qualified
    out['normal_min'][~qualified]=0.;out['normal_max'][~qualified]=0.
    return out

def observe(points, owners, frames, frame_available, records, cache=None):
    meshes=[r['mesh']for r in records]
    """Owner and two distinct nonowner distance layers, with tie envelopes."""
    points, frames = np.asarray(points, float), np.asarray(frames, float)
    owners, frame_available = np.asarray(owners, int), np.asarray(frame_available, bool)
    known = (owners >= 0) & (owners < len(meshes))
    values = np.zeros((len(points), len(NAMES)), np.float32)
    values[:, :3] = np.column_stack([np.ones(len(points)), known, frame_available])
    cache = {} if cache is None else cache
    boxes = np.asarray([r['box'] for r in records])
    body_valid=np.asarray([r['solid_valid']for r in records],bool)
    values[:,28]=bool(body_valid.all())
    lower = np.linalg.norm(np.maximum(np.maximum(boxes[:, :3][None]-points[:, None],
                                                points[:, None]-boxes[:, 3:][None]), 0), axis=2)
    distance = np.full((len(points), len(meshes)), np.inf)
    available = np.zeros(distance.shape, bool)
    low, high = np.zeros((*distance.shape, 3)), np.zeros((*distance.shape, 3))
    second = np.full(len(points), np.inf)
    queries = 0
    for owner in np.unique(owners[known]):
        indices = np.flatnonzero(owners == owner)
        part = closest(points[indices], records[owner], frames[indices], cache)
        queries += len(indices)
        valid = part['normal_available'] & frame_available[indices]
        values[indices, 3] = np.log1p(np.where(np.isfinite(part['distance_mm']),part['distance_mm'],0.))
        values[indices,25]=bool(body_valid[owner])
        values[indices, 4] = valid
        for axis in range(3):
            values[indices, 5+2*axis] = np.where(valid, part['normal_min'][:, axis], 0)
            values[indices, 6+2*axis] = np.where(valid, part['normal_max'][:, axis], 0)
    for mi in np.argsort(np.median(lower, axis=0), kind='stable'):
        mi = int(mi)
        take = np.flatnonzero(known & (owners != mi) & (lower[:, mi] <= second+LAYER_TIE_MM+indexed.PRUNE_GUARD_MM))
        if not len(take):
            continue
        part = closest(points[take], records[mi], frames[take], cache)
        queries += len(take)
        distance[take, mi], available[take, mi] = part['distance_mm'], part['normal_available']
        low[take, mi], high[take, mi] = part['normal_min'], part['normal_max']
        first = distance.min(1)
        second = np.where(distance > first[:, None]+LAYER_TIE_MM, distance, np.inf).min(1)
    first = distance.min(1)
    for li, target in enumerate((first, second)):
        finite = np.isfinite(distance) & np.isfinite(target)[:, None]
        difference = np.full(distance.shape, np.inf)
        np.subtract(distance, target[:, None], out=difference, where=finite)
        tied = finite & (np.abs(difference) <= LAYER_TIE_MM)
        valid = tied & available & frame_available[:, None]
        has = valid.any(1) & ~(tied & ~available).any(1)
        # A tied unknown face is retained as uncertainty, never discarded.
        for mi,record in enumerate(records):
            take=known & (owners!=mi)
            if record['missing_boxes'].size:
                uncertain=box_lower(points,record['missing_boxes'])<=target+SUPPORT_BAND_MM+TIE_MM
                has &= ~(take & uncertain)
        finite_target=np.isfinite(target)
        values[:,26+li]=finite_target & ~(tied & ~body_valid[None,:]).any(1)
        values[:,29+2*li]=finite_target
        values[:,30+2*li]=np.log1p(np.where(finite_target,target,0.))
        offset = 11+li*7
        values[:, offset] = has
        for axis in range(3):
            a = np.where(valid, low[:, :, axis], np.inf).min(1)
            b = np.where(valid, high[:, :, axis], -np.inf).max(1)
            values[:, offset+1+2*axis] = np.where(has, a, 0)
            values[:, offset+2+2*axis] = np.where(has, b, 0)
    assert np.isfinite(values).all()
    return values, dict(owner_known=known.tolist(), exact_triangle_queries=queries,
        nearest_distinct_layer_distances_mm=[np.where(np.isfinite(x), x, -1).tolist() for x in (first, second)],
        unavailable_distance_minus1_is_receipt_only_not_model_input=True,
        mesh_normal_band_mm=SUPPORT_BAND_MM, no_weld_or_process_decision=True)


def prepare_meshes(packet,solids):
    cad=original.cad;transform=np.asarray(packet['pose']['transform_STEP_to_XYZ'])
    assert packet['pose']['passed'] and transform.shape==(4,4) and solids
    basis=shadow.basis(packet['base_axis']);origin=transform[:3,3];records=[]
    def local_box(shape):
        box=cad.bbox(shape)
        if not np.isfinite(box).all():return np.r_[np.full(3,-1e30),np.full(3,1e30)]
        corners=np.asarray(list(itertools.product(*zip(box[:3],box[3:]))))
        corners=((corners@transform[:3,:3].T+transform[:3,3])-origin)@basis
        return np.r_[corners.min(0),corners.max(0)]
    for solid in solids:
        volume_valid=bool(BRepCheck_Analyzer(solid,True).IsValid())
        cad.BRepTools.Clean_s(solid);cad.BRepMesh_IncrementalMesh(solid,DEFLECTION_MM,False,.2,False)
        chunks=[];valid=[];missing=[];invalid_ids=[];unmeshed_ids=[];faces=list(cad.items(solid,cad.TopAbs_FACE))
        for fi,item in enumerate(faces):
            face=cad.TopoDS.Face_s(item);qualified=bool(BRepCheck_Analyzer(face,True).IsValid())
            if not qualified:invalid_ids.append(fi)
            location=cad.TopLoc_Location();mesh=cad.BRep_Tool.Triangulation_s(face,location)
            if mesh is None or mesh.NbTriangles()==0:
                missing.append(local_box(face));unmeshed_ids.append(fi);continue
            nodes=np.asarray([cad.vec(mesh.Node(i).Transformed(location.Transformation()))for i in range(1,mesh.NbNodes()+1)])
            ids=np.asarray([mesh.Triangle(i).Get()for i in range(1,mesh.NbTriangles()+1)],int)-1
            triangle=nodes[ids];triangle=((triangle@transform[:3,:3].T+transform[:3,3])-origin)@basis
            if not np.isfinite(triangle).all():
                missing.append(local_box(face));unmeshed_ids.append(fi);continue
            chunks.append(triangle);valid.extend([qualified]*len(triangle))
        triangles=np.concatenate(chunks)if chunks else np.empty((0,3,3))
        mask=np.asarray(valid,bool)
        records.append(dict(mesh=triangles,bad_mesh=triangles[~mask],missing_boxes=np.asarray(missing).reshape(-1,6),
            box=local_box(solid),solid_valid=volume_valid,faces=len(faces),invalid_face_ids=invalid_ids,unmeshed_face_ids=unmeshed_ids,
            qualified_triangles=int(mask.sum()),total_triangles=len(triangles)))
    return records,basis,origin


def build(packet, solids):
    start = time.perf_counter()
    values = np.zeros((packet['offsets'][-1], len(NAMES)), np.float32)
    pose = packet.get('pose', {})
    if not (solids and pose.get('passed') and np.asarray(pose.get('transform_STEP_to_XYZ', [])).shape == (4, 4)):
        return values, dict(available=False, reason='NO_QUALIFIED_FINITE_STEP', geometry_and_GT_retained=True,
                            seconds=time.perf_counter()-start, names=list(NAMES))
    records, basis, origin = prepare_meshes(packet, solids)
    owner_at = indexed.owner_function(packet)
    cache, details = {}, []
    for ci, candidate in enumerate(packet['candidates']):
        lo, hi = packet['offsets'][ci:ci+2]
        points = (np.asarray(candidate['stations'])-origin)@basis
        native, receipt = reference.parameters(candidate)
        owners = np.asarray([owner_at(ci, float(s), packet) for s in native])
        closed = np.linalg.norm(np.asarray(candidate['points'])[0]-candidate['points'][-1]) <= 1e-6
        frames, frame_available = direction.station_frames(points, closed)
        part, observed = observe(points, owners, frames, frame_available, records, cache)
        values[lo:hi] = part
        details.append(dict(candidate=ci, native_reference=receipt, observation=observed, geometry_retained=True))
    return values, dict(available=True, names=list(NAMES), candidates=details, materials=len(solids), material_surface_qualification=[{k:v for k,v in r.items()if k not in ('mesh','bad_mesh','missing_boxes','box')}for r in records],
        mesh_triangles=sum(len(r['mesh']) for r in records), mesh_deflection_mm=DEFLECTION_MM,
        unsigned_normals_not_oriented_contract_attributes=True, all_compatible_triangle_normals_enveloped=True,
        mesh_band_not_weld_clearance_or_matching_tolerance=True, no_labels_models_or_process_decision=True,
        seconds=time.perf_counter()-start)
