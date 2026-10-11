"""Separate endpoint geometry tolerance from gaps between surface samples.

Interpolated boundaries and sampled triangles remain material hypotheses.
Neither a hidden hole nor solid identity is inferred from a missing sample.
"""
import numpy as np
from scipy.spatial import Delaunay, QhullError

from weld_pipeline.finite_joint_base_support import FiniteJointBases, unit
from weld_pipeline.local_base_sampling import sampling_fill_radius

VERSION='continuous_contact_and_base_sampling_v1'
GEOMETRY_TOLERANCE_MM=3.


def closest_segment(q,segments):
    a=segments[:,0];d=segments[:,1]-a
    denominator=np.einsum('ij,ij->i',d,d)
    ratio=np.clip(np.einsum('ij,ij->i',q-a,d)/np.maximum(denominator,1e-12),0.,1.)
    closest=a+ratio[:,None]*d
    distances=np.linalg.norm(closest-q,axis=1)
    index=int(np.argmin(distances))
    return closest[index],float(distances[index])


def boundary_projection(plane,point,inward):
    q=np.asarray(point,float);basis=plane['basis'];n=plane['normal']
    projected=q+n*(plane['offset']-q@n);centre=projected@basis.T
    u=np.asarray(inward,float)@basis.T
    if np.linalg.norm(u)<1e-8:
        return None,dict(reason='DEGENERATE_CONTACT_BOUNDARY_DIRECTION')
    u/=np.linalg.norm(u);v=np.array([-u[1],u[0]])
    ids=plane['tree'].query_ball_point(centre,48.)
    relative=(plane['uv'][ids]-centre)@np.stack((u,v)).T
    candidates=[]
    nearest,index=plane['tree'].query(centre)
    sample_uv=plane['uv'][int(index)]
    sample=sample_uv@basis+plane['offset']*n
    sample_distance=float(np.linalg.norm(sample-q))
    if sample_distance<=GEOMETRY_TOLERANCE_MM:
        candidates.append((sample_distance,sample,dict(source='actual_surface_sample',nearest_sample_mm=float(nearest))))
    if len(relative):
        bins=np.round(relative[:,1]/.15).astype(np.int64)
        for row in np.unique(bins[np.abs(relative[:,1])<=GEOMETRY_TOLERANCE_MM]):
            positions=relative[bins==row]
            order=np.argsort(positions[:,0]);positions=positions[order]
            _,unique=np.unique(np.round(positions[:,0],5),return_index=True)
            positions=positions[np.sort(unique)]
            if len(positions)<4:
                continue
            gaps=np.diff(positions[:,0]);positive=gaps[gaps>1e-4]
            if len(positive)<3:
                continue
            spacing=float(np.median(positive))
            before=np.flatnonzero(positions[:,0]<=0.);after=np.flatnonzero(positions[:,0]>=0.)
            if not len(before) or not len(after):
                continue
            a,b=positions[before[-1]],positions[after[0]]
            gap=float(np.linalg.norm(b-a))
            if gap>1.8*spacing:
                continue
            local,distance=closest_segment(np.zeros(2),np.array([[a,b]]))
            boundary_uv=centre+local[0]*u+local[1]*v
            boundary=boundary_uv@basis+plane['offset']*n
            geometry_distance=float(np.linalg.norm(boundary-q))
            if geometry_distance<=GEOMETRY_TOLERANCE_MM:
                candidates.append((geometry_distance,boundary,dict(source='locally_regular_sampled_boundary',
                    boundary_sample_gap_mm=gap,local_typical_gap_mm=spacing,
                    endpoint_to_boundary_in_plane_mm=distance,nearest_sample_mm=float(nearest),
                    boundary_points=len(positions),hidden_hole_continuity_verified=False)))
    if not candidates:
        return None,dict(reason='NO_CONTINUOUS_CONTACT_WITHIN_GEOMETRY_TOLERANCE',nearest_sample_mm=float(nearest))
    candidates.sort(key=lambda item:item[0])
    distance,position,evidence=candidates[0]
    return position,dict(geometry_distance_mm=distance,**evidence,material_identity_verified=False)


def project_contact(profile,bases,point,normal,inward):
    q=np.asarray(point,float);choices=[];attempts=[]
    for plane in bases.candidates(q,normal):
        position,evidence=boundary_projection(plane,q,inward)
        attempts.append(evidence)
        if position is not None:
            choices.append((evidence['geometry_distance_mm'],position,evidence))
    if not choices:
        return None,dict(reason='NO_CONTINUOUS_FINITE_CONTACT_WITHIN_GEOMETRY_TOLERANCE',attempts=attempts)
    choices.sort(key=lambda item:item[0])
    if len(choices)>1 and choices[1][0]-choices[0][0]<=.15:
        return None,dict(reason='MULTIPLE_CONTINUOUS_CONTACT_PLANES_NEAR_ENDPOINT',attempts=attempts)
    distance,projected,evidence=choices[0]
    features,observation=profile.observe(projected,normal)
    if features[0]==0 or observation['component'] is None:
        return None,dict(reason='FINITE_CONTACT_COMPONENT_UNRESOLVED',projection=evidence)
    return (features,observation,projected),dict(point_projection_mm=projected.tolist(),
        geometry_correction_mm=distance,sample_continuity=evidence,
        finite_material_identity_verified=False)


def patch_projection(centre,triangulation,supported,local):
    simplex=int(triangulation.find_simplex(centre))
    if simplex>=0 and supported[simplex]:
        return centre.copy(),0.
    triangles=local[triangulation.simplices[supported]]
    if not len(triangles):
        return None,None
    segments=np.stack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]),axis=1).reshape(-1,2,2)
    return closest_segment(centre,segments)


class ContinuousJointBases(FiniteJointBases):
    def support(self,point,base_normal,toward_interior):
        q=np.asarray(point,float);attempts=[]
        for plane in self.candidates(q,base_normal):
            n,basis,offset,tree=(plane[k] for k in ('normal','basis','offset','tree'))
            projected=q+n*(offset-q@n);centre=projected@basis.T
            inward3=np.asarray(toward_interior,float);inward3=inward3-n*(inward3@n)
            if np.linalg.norm(inward3)<1e-8:
                return None,dict(reason='DEGENERATE_CHILD_IN_BASE_PLANE')
            inward=unit(inward3)@basis.T;lateral=np.array([-inward[1],inward[0]])
            ids=tree.query_ball_point(centre,48.)
            failure=dict(plane_normal=n.tolist(),plane_offset_mm=offset,
                endpoint_to_plane_mm=float(abs(q@n-offset)),closest_joint_sample_mm=float(tree.query(centre)[0]),
                local_points=len(ids))
            if len(ids)<9:
                attempts.append({**failure,'reason':'INSUFFICIENT_FINITE_BASE_SAMPLES'});continue
            local=tree.data[ids];relative=(local-centre)@np.stack((inward,lateral)).T
            if len(np.unique(np.round(relative[:,1]/.15)))<3:
                attempts.append({**failure,'reason':'BASE_INTERIOR_ONLY_OUTLINE_ROWS'});continue
            _,sampling=sampling_fill_radius(tree,centre,inward,lateral)
            if sampling['state']!='LOCAL_2D_SAMPLING_GAPS_MEASURED':
                attempts.append({**failure,'reason':sampling['state']});continue
            try:
                triangulation=Delaunay(local)
            except QhullError:
                attempts.append({**failure,'reason':'NO_TWO_DIMENSIONAL_BASE_TRIANGULATION'});continue
            triangles=local[triangulation.simplices]
            edges=np.stack((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,1],triangles[:,0]-triangles[:,2]),axis=1)
            scaled=np.stack((edges@inward/sampling['inward_spacing_mm'],edges@lateral/sampling['lateral_spacing_mm']),axis=-1)
            supported=np.linalg.norm(scaled,axis=-1).max(1)<=1.8
            contact,planar_distance=patch_projection(centre,triangulation,supported,local)
            if contact is None or np.hypot(planar_distance,q@n-offset)>GEOMETRY_TOLERANCE_MM:
                attempts.append({**failure,'reason':'NO_CONTINUOUS_BASE_WITHIN_GEOMETRY_TOLERANCE'});continue
            accepted=None
            for side in (-1.,1.):
                for distances in ((6.,18.),(2.,6.)):
                    probes=np.array([contact+u*inward+side*v*lateral for u in (6.,18.) for v in distances])
                    simplices=triangulation.find_simplex(probes);inside=simplices>=0;good=inside.copy()
                    good[inside]&=supported[simplices[inside]]
                    if good.all():
                        accepted=dict(material_side=side,lateral_probe_mm=list(distances),
                                      probes_world_mm=(probes@basis+offset*n).tolist());break
                if accepted is not None:
                    break
            if accepted is None:
                attempts.append({**failure,'reason':'LOCAL_BASE_MATERIAL_PROBES_NOT_SUPPORTED','sampling':sampling});continue
            projection=contact@basis+offset*n
            return dict(version=VERSION,plane_normal=n.tolist(),plane_offset_mm=offset,
                projected_joint_mm=projection.tolist(),local_base_z_mm=float(projection[2]),sampling=sampling,
                support=accepted,local_points=len(ids),endpoint_to_supported_surface_mm=float(np.linalg.norm(projection-q)),
                nearest_sample_mm=failure['closest_joint_sample_mm'],reason='SAMPLED_CONTINUOUS_BASE_HYPOTHESIS',
                material_identity_verified=False,hidden_holes_ruled_out=False,reviewed_customer_truth=False),dict(attempts=attempts)
        return None,dict(reason='FINITE_CONTINUOUS_JOINT_BASE_NOT_RESOLVED',attempts=attempts)
