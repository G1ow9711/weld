"""Point-only horizontal arc contacts. These are unclassified candidates."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.local_tangent_circles import propose_local_tangent_circles
from weld_pipeline.circle_candidates import fit_xy_circle


def supported_arc(xyz, proposal, radial_tolerance_mm=.15):
    """Use all surface samples, including the cylinder's top/bottom boundaries."""
    center=np.asarray(proposal['center_xy_mm'],float)
    radius=float(proposal['radius_mm'])
    hits=xyz[np.abs(np.linalg.norm(xyz[:,:2]-center,axis=1)-radius)<=radial_tolerance_mm]
    xy=np.unique(np.round(hits[:,:2],3),axis=0)
    if len(xy)<8:return None
    fitted=fit_xy_circle(xy)
    if fitted is None:return None
    center,radius,residual=fitted
    if residual.max()>radial_tolerance_mm:return None
    theta=np.sort(np.arctan2(xy[:,1]-center[1],xy[:,0]-center[0])%(2*np.pi))
    gaps=np.diff(np.r_[theta,theta[0]+2*np.pi])
    cut=int(np.argmax(gaps));start=float(theta[(cut+1)%len(theta)])
    extent=float(2*np.pi-gaps[cut]);inside=np.delete(gaps,cut)
    # Scattered intersections with unrelated walls do not trace an arc.
    # Preserve supported curved surfaces; do not infer full circles from three points.
    if extent<np.deg2rad(10) or extent>np.deg2rad(180):return None
    if inside.max()>max(np.deg2rad(15),3*np.median(inside)):return None
    if np.ptp(hits[:,2])<5:return None
    return {'center_xy_mm':center.tolist(),'radius_mm':float(radius),
            'start_angle_rad':start,'sweep_rad':extent,'xy_support':len(xy),
            'z_range_mm':[float(hits[:,2].min()),float(hits[:,2].max())],
            'radial_max_error_mm':float(residual.max()),
            'max_interior_angle_gap_degrees':float(np.degrees(inside.max()))}


def sample_arc(arc, height_mm, step_mm=1.):
    if not np.isfinite(step_mm) or step_mm<=0:raise ValueError('Positive finite arc step required')
    length=arc['radius_mm']*arc['sweep_rad']
    theta=arc['start_angle_rad']+np.linspace(0,arc['sweep_rad'],max(2,int(np.ceil(length/step_mm))+1))
    xy=np.asarray(arc['center_xy_mm'])+arc['radius_mm']*np.column_stack([np.cos(theta),np.sin(theta)])
    return np.column_stack([xy,np.full(len(xy),height_mm)])


def plane_boundary_arc(arc,plane):
    """Refine to an observed planar hole edge, without a fixed weld offset."""
    center=np.asarray(arc['center_xy_mm']);delta=plane[:,:2]-center
    radii=np.linalg.norm(delta,axis=1)
    angle=(np.arctan2(delta[:,1],delta[:,0])-arc['start_angle_rad'])%(2*np.pi)
    inside=(angle<=arc['sweep_rad']+.01)|(angle>=2*np.pi-.01)
    mask=inside&(np.abs(radii-arc['radius_mm'])<=5.)
    xy=np.unique(np.round(plane[mask,:2],3),axis=0)
    if len(xy)<8:return {**arc,'plane_boundary_refined':False}
    radial=np.linalg.norm(xy-center,axis=1);bins,counts=np.unique(np.round(radial,1),return_counts=True)
    for index in np.argsort(counts)[::-1]:
        if counts[index]<8:break
        p=xy[np.abs(radial-bins[index])<=.06]
        fit=fit_xy_circle(p)
        if fit is None:continue
        refined,radius,error=fit
        if error.max()>.06 or np.linalg.norm(refined-center)>1.:continue
        angles=(np.arctan2(p[:,1]-refined[1],p[:,0]-refined[0])-arc['start_angle_rad']+np.pi)%(2*np.pi)-np.pi
        # This prototype's supported sweep is at most pi, so choose its branch.
        angles[angles<-.05]+=2*np.pi
        if np.ptp(angles)<.8*arc['sweep_rad']:continue
        return {**arc,'center_xy_mm':refined.tolist(),'radius_mm':float(radius),
                'start_angle_rad':float(arc['start_angle_rad']+angles.min()),
                'sweep_rad':float(np.ptp(angles)),'plane_boundary_refined':True,
                'boundary_xy_support':len(p),'boundary_radial_max_error_mm':float(error.max()),
                'cylinder_radius_mm':arc['radius_mm']}
    return {**arc,'plane_boundary_refined':False}


def arc_surface_candidates(xyz, step_mm=1.):
    xyz=np.asarray(xyz,dtype=float)
    proposals,raw_stats=propose_local_tangent_circles(xyz)
    arcs=[a for c in proposals if (a:=supported_arc(xyz,c)) is not None]
    if not len(xyz):return [],{'cylinders':raw_stats,'supported_arcs':0,'horizontal_levels_mm':[]}
    levels,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    levels=levels[counts>=max(50,len(xyz)*.01)]
    trees={};planes={};result=[]
    for ai,arc in enumerate(arcs):
        for level in levels:
            if not arc['z_range_mm'][0]-.05<=level<=arc['z_range_mm'][1]+.05:continue
            if float(level) not in trees:
                plane=xyz[np.abs(xyz[:,2]-level)<.015]
                planes[float(level)]=plane
                trees[float(level)]=cKDTree(plane)
            contact=plane_boundary_arc(arc,planes[float(level)])
            q=sample_arc(contact,float(level),step_mm)
            distance=trees[float(level)].query(q,workers=2)[0]
            if np.mean(distance<=5.)<.8:continue
            # A sampled rim alone is not evidence of a horizontal plate.
            radial=np.column_stack([(q[:,:2]-np.asarray(contact['center_xy_mm']))/contact['radius_mm'],np.zeros(len(q))])
            fractions=[]
            for offset in [5.,15.,30.]:
                sides=q[:,None,:]+radial[:,None,:]*np.array([-offset,offset])[None,:,None]
                side_distances=trees[float(level)].query(sides.reshape(-1,3),workers=2)[0].reshape(-1,2)
                fractions.append(float(np.mean(side_distances.min(1)<=offset*.6)))
            plate_fraction=max(fractions)
            if plate_fraction<.8:continue
            result.append({'id':f'arc_candidate_{len(result)+1:04d}','type':'horizontal',
                           'segments':[{'kind':'arc','points':q.tolist(),
                                        'length_mm':float(contact['radius_mm']*contact['sweep_rad'])}],
                           'circle_index':ai,'arc':contact,'height_mm':float(level),
                           'plane_support_fraction_at_5mm':float(np.mean(distance<=5.)),
                           'plate_side_support_fraction':plate_fraction})
    return result,{'cylinders':raw_stats,'supported_arcs':len(arcs),
                   'horizontal_levels_mm':levels.tolist(),'arc_contact_candidates':len(result),
                   'scope':'Unclassified contact curves; not predicted welds or process segmentation.'}
