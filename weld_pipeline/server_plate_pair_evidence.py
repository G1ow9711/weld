"""Original local two-plate evidence, reusable by complete region recovery.

Plane, opposite-face, coverage and signed-normal checks are preserved from the
reviewed geometry implementation. Geometric evidence is not welding truth or
customer-certified material identity. Historical promotion/output gates are
not included here.
"""
import numpy as np
from scipy.spatial import cKDTree

POLICY=dict(stations=[.2,.5,.8],section_half_width_mm=10.,transverse_min_mm=5.,transverse_max_mm=80.,
    opposite_search_max_mm=150.,minimum_thickness_mm=1.,plane_p95_max_mm=.05,
    parallel_angle_max_deg=.1,opposite_coverage_min=.8,coverage_distance_mm=3.,
    minimum_patch_points=30,minimum_width_to_thickness=2.,minimum_good_stations=2,
    maximum_uniform_thickness_spread_mm=.25)

def fit(points, initial_normal):
    if len(points)<POLICY['minimum_patch_points']:return None
    p=points;normal=initial_normal
    for _ in range(4):
        center=p.mean(0);d=p-center;e,v=np.linalg.eigh(d.T@d/len(p));n=v[:,0]
        if n@initial_normal<0:n=-n
        if abs(n@initial_normal)<np.cos(np.deg2rad(.5)) or e[1]<1.:return None
        residual=abs((points-center)@n);median=np.median(residual);mad=np.median(abs(residual-median))
        keep=residual<=max(.02,min(.3,median+6*mad))
        if keep.sum()<max(POLICY['minimum_patch_points'],len(points)//2):return None
        p=points[keep];normal=n
    residual=abs((p-center)@normal);error=float(np.quantile(residual,.95))
    if error>POLICY['plane_p95_max_mm']:return None
    return dict(points=p,center=center,normal=normal,p95_mm=error)

def one_plate(points, origin, direction, normal):
    normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
    transverse=np.cross(normal,direction);norm=np.linalg.norm(transverse)
    if norm<.1:return dict(status='line_not_tangent_to_face')
    transverse/=norm;delta=points-origin;v=delta@transverse;w=delta@normal
    mask=(abs(v)>=POLICY['transverse_min_mm'])&(abs(v)<=POLICY['transverse_max_mm'])
    cloud=points[mask];signed=w[mask]
    own=fit(cloud[abs(signed)<=.75],normal)
    if own is None:return dict(status='no_stable_planar_face')
    normal=own['normal'];signed=(cloud-own['center'])@normal
    far=(abs(signed)>=POLICY['minimum_thickness_mm'])&(abs(signed)<=POLICY['opposite_search_max_mm'])
    if far.sum()<POLICY['minimum_patch_points']:return dict(status='no_opposite_face')
    bins,counts=np.unique(np.rint(signed[far]/.1).astype(np.int64),return_counts=True)
    tested=[];candidates=[]
    own_v=(own['points']-origin)@transverse
    width=float(max(0.,np.quantile(own_v,.95))-min(0.,np.quantile(own_v,.05)))
    for i in np.argsort(-counts)[:20]:
        distance=float(bins[i])*.1
        if any(abs(distance-d)<.4 for d in tested):continue
        tested.append(distance)
        opposite=fit(cloud[abs(signed-distance)<=.2],normal)
        if opposite is None:continue
        if abs(opposite['normal']@normal)<np.cos(np.deg2rad(POLICY['parallel_angle_max_deg'])):continue
        signed_thickness=float((opposite['center']-own['center'])@normal);thickness=abs(signed_thickness)
        if not POLICY['minimum_thickness_mm']<=thickness<=POLICY['opposite_search_max_mm']:continue
        if width < POLICY['minimum_width_to_thickness']*thickness:continue
        projected=opposite['points']-signed_thickness*normal
        distances=cKDTree(projected).query(own['points'],workers=2)[0]
        coverage=float(np.mean(distances<=POLICY['coverage_distance_mm']))
        if coverage<POLICY['opposite_coverage_min']:continue
        outward=-np.sign(signed_thickness)*normal
        candidates.append(dict(thickness_mm=thickness,directed_normal=outward.tolist(),opposite_coverage=coverage,
            face_p95_mm=own['p95_mm'],opposite_p95_mm=opposite['p95_mm'],observed_transverse_width_mm=width,
            own_points=len(own['points']),opposite_points=len(opposite['points']),
            face_offset_from_prediction_mm=float((own['center']-origin)@normal),
            face_center_mm=own['center'].tolist(),opposite_face_center_mm=opposite['center'].tolist(),
            face_fit_samples_mm=own['points'][np.linspace(0,len(own['points'])-1,min(128,len(own['points'])),dtype=int)].tolist(),
            directed_plane_offset_mm=float(own['center']@outward)))
    if not candidates:return dict(status='no_parallel_opposite_with_matching_footprint')
    # Multiple supported opposite faces are ambiguous; do not choose a thickness
    # merely because it is convenient or closer to a presumed stock dimension.
    distinct=[]
    for c in sorted(candidates,key=lambda v:-v['opposite_coverage']):
        if all(abs(c['thickness_mm']-p['thickness_mm'])>.4 or np.dot(c['directed_normal'],p['directed_normal'])<.99 for p in distinct):distinct.append(c)
    if len(distinct)>1:return dict(status='ambiguous_parallel_faces',options=distinct)
    return dict(status='SUPPORTED_PARALLEL_PLATE',**distinct[0])

def segment_attributes(xyz,tree,segment,normals):
    ends=np.asarray(segment['endpoints_mm'],float);direction=ends[1]-ends[0];length=np.linalg.norm(direction)
    if length<=1e-6:return dict(status='degenerate_line',plate_pair_supported=False)
    direction/=length;stations=[]
    for fraction in POLICY['stations']:
        origin=ends[0]+fraction*(ends[1]-ends[0]);ids=tree.query_ball_point(origin,175.,workers=2)
        points=np.asarray(xyz[ids]);delta=points-origin
        points=points[abs(delta@direction)<=POLICY['section_half_width_mm']]
        faces=[one_plate(points,origin,direction,n) for n in normals]
        stations.append(dict(fraction=fraction,position_mm=origin.tolist(),faces=faces,
            both_supported=all(f['status']=='SUPPORTED_PARALLEL_PLATE' for f in faces)))
    good=[s for s in stations if s['both_supported']]
    if len(good)<POLICY['minimum_good_stations']:
        return dict(status='INSUFFICIENT_TWO_PLATE_EVIDENCE',plate_pair_supported=False,stations=stations)
    plates=[]
    for side in range(2):
        thickness=np.array([s['faces'][side]['thickness_mm'] for s in good])
        normals=np.array([s['faces'][side]['directed_normal'] for s in good]);reference=normals[0]
        if np.min(normals@reference)<np.cos(np.deg2rad(1.)):
            return dict(status='INCONSISTENT_MATERIAL_SIDE',plate_pair_supported=False,stations=stations)
        normal=normals.mean(0);normal/=np.linalg.norm(normal)
        # A long line needs a plane fitted across its full span. Averaging local
        # small-patch normals can introduce artificial long-range gaps.
        samples=np.vstack([s['faces'][side]['face_fit_samples_mm'] for s in good])
        global_plane=fit(samples,normal)
        if global_plane is None:return dict(status='GLOBAL_PLANE_NOT_STABLE',plate_pair_supported=False,stations=stations)
        normal=global_plane['normal'];center=global_plane['center']
        thickness=np.array([abs((np.asarray(s['faces'][side]['opposite_face_center_mm'])-np.asarray(s['faces'][side]['face_center_mm']))@normal) for s in good])
        plates.append(dict(thickness_mm=float(np.median(thickness)),directed_normal=normal.tolist(),
            directed_plane_offset_mm=float(center@normal),global_face_p95_mm=global_plane['p95_mm'],
            thickness_spread_mm=float(np.ptp(thickness)),normal_unit_error=float(abs(np.linalg.norm(normal)-1.)),
            evidence='opposite parallel surface and local overlapping footprint; not verified truth'))
    varying=any(p['thickness_spread_mm']>POLICY['maximum_uniform_thickness_spread_mm'] for p in plates)
    return dict(status='LOCAL_THICKNESS_VARIES_REQUIRES_SEGMENT_REVIEW' if varying else 'SUPPORTED_TWO_PLATE_GEOMETRY',
        plate_pair_supported=True,uniform_thickness=not varying,plates=plates,stations=stations,
        material_identity_proven=False,welding_process_proven=False,truth_accuracy_verified=False)

def self_test():
    a=np.arange(-60.,61.,2.);b=np.arange(-50.,51.,2.);y,x=np.meshgrid(a,b)
    floor=np.vstack([np.c_[x.ravel(),y.ravel(),np.full(x.size,z)] for z in [0.,-10.]])
    y,z=np.meshgrid(a,np.arange(0.,81.,2.));stem=np.vstack([np.c_[np.full(y.size,xv),y.ravel(),z.ravel()] for xv in [-5.,5.]])
    points=np.vstack([floor,stem]);segment=dict(endpoints_mm=[[5.,-40.,0.],[5.,40.,0.]])
    result=segment_attributes(points,cKDTree(points),segment,[np.array([1.,0.,0.]),np.array([0.,0.,1.])])
    assert result['plate_pair_supported'],result
    assert all(abs(p['thickness_mm']-10.)<1e-8 for p in result['plates'])
    assert np.allclose(result['plates'][0]['directed_normal'],[1,0,0]) and np.allclose(result['plates'][1]['directed_normal'],[0,0,1])
    # Rigid translation/rotation must preserve thickness and rotate normals.
    angle=.47;r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    shift=np.array([50000.,-40000.,21000.]);moved=points@r.T+shift;ends=np.asarray(segment['endpoints_mm'])@r.T+shift
    check=segment_attributes(moved,cKDTree(moved),dict(endpoints_mm=ends),[r[:,0],r[:,2]])
    assert check['plate_pair_supported'] and all(abs(p['thickness_mm']-10.)<1e-7 for p in check['plates'])
    assert np.allclose(check['plates'][0]['directed_normal'],r[:,0],atol=1e-8)
    # A single unpaired plane must not acquire an invented thickness.
    single=one_plate(floor[floor[:,2]==0],np.zeros(3),np.array([0.,1.,0.]),np.array([0.,0.,1.]))
    assert single['status']!='SUPPORTED_PARALLEL_PLATE'
    # The end face of one flat plate must not count as a second thin plate.
    y,z=np.meshgrid(a,np.arange(-10.,.1,2.));ends=np.vstack([np.c_[np.full(y.size,xv),y.ravel(),z.ravel()] for xv in [-50.,50.]])
    closed=np.vstack([floor,ends]);edge=segment_attributes(closed,cKDTree(closed),dict(endpoints_mm=[[50.,-40.,0.],[50.,40.,0.]]),[np.array([1.,0.,0.]),np.array([0.,0.,1.])])
    assert not edge['plate_pair_supported'],edge
    print('PLATE_CONTROLS_PASSED: opposite faces, directed normals, rigid transform, no invented thickness',flush=True)

ORIGINAL_REFERENCE_SHA256 = 'bc9d1faf00bbaaf15041b43717502e887805577f652ea86b44337b3e57fc3903'
ORIGINAL_EVIDENCE_FUNCTION_BODIES_SHA256 = '12d9ab2ef5d744215b3ae5c3bd776f0ccf045334d1134591b413e7d2f2845b37'
