"""Separate ideal-CAD plane support from neighboring surfaces in the fit band.

The 0.2 mm band remains the initial discovery tolerance. Observed robust plane
residuals determine the final support, without using CAD or reference endpoints.
Noisy scan applicability remains unvalidated.
"""
import numpy as np
from weld_pipeline.sjg_native_finite_geometry_982 import extract as original_extract
from weld_pipeline.sjg_union_sampling_mesh_995 import local_support


def refine_support(xyz,plane):
    ids=plane['point_ids'];points=xyz[ids]
    normal=np.asarray(plane['normal']);offset=float(plane['offset'])
    center=np.asarray(plane['center']);basis=np.asarray(plane['basis'])
    for iteration in range(3):
        residual=np.abs(points@normal-offset);median=float(np.median(residual))
        mad=float(np.median(np.abs(residual-median)))
        tolerance=min(.2,max(1e-4,median+6.*mad))
        keep=residual<=tolerance
        if keep.sum()<max(16,len(ids)//2):
            return dict(plane,ideal_plane_refinement_status='INSUFFICIENT_DOMINANT_SUPPORT')
        core=points[keep];center=core.mean(0);delta=core-center
        eigen,vectors=np.linalg.eigh(delta.T@delta/len(core))
        if eigen[1]<=1e-8:return dict(plane,ideal_plane_refinement_status='RANK_DEFICIENT')
        normal=vectors[:,0]
        if normal[np.argmax(abs(normal))]<0:normal=-normal
        offset=float(center@normal);basis=vectors[:,1:]
    residual=np.abs(points@normal-offset);median=float(np.median(residual));mad=float(np.median(abs(residual-median)))
    tolerance=min(.2,max(1e-4,median+6.*mad));keep=residual<=tolerance
    return dict(plane,normal=normal,offset=offset,center=center,basis=basis,point_ids=ids[keep],
        ideal_plane_refinement_status='ROBUST_OBSERVED_SUPPORT',initial_discovery_tolerance_mm=.2,
        adaptive_support_tolerance_mm=tolerance,removed_nearby_surface_points=int((~keep).sum()),
        observed_residual_median_mm=median,observed_residual_MAD_mm=mad,
        discarded_points_remain_in_original_denominator=True,noisy_scan_applicability_unvalidated=True)


def extract(xyz,pred,instance,material,progress=None):
    return original_extract(xyz,pred,instance,material,progress,mesh_builder=local_support,
        preserve_cell_extrema=True,shared_membership=True,preserve_predicted_boundary=True,support_refiner=refine_support)


def controls():
    from weld_pipeline.sjg_union_sampling_mesh_995 import controls as mesh_controls
    result=mesh_controls()
    x,y=np.meshgrid(np.arange(-20.,21.),np.arange(-20.,21.))
    exact=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    pollution=exact[::10]+np.array([0.,0.,.077])
    xyz=np.r_[exact,pollution]
    for angle in (0.,.71):
        rotation=np.array([[np.cos(angle),0.,np.sin(angle)],[0.,1.,0.],[-np.sin(angle),0.,np.cos(angle)]])
        points=xyz@rotation.T+np.array([1000.,-700.,250.])
        n=rotation[:,2];center=points.mean(0)
        plane=dict(point_ids=np.arange(len(points)),normal=n,center=center,offset=float(center@n),basis=rotation[:,:2])
        refined=refine_support(points,plane)
        assert np.array_equal(refined['point_ids'],np.arange(len(exact)))
        assert abs(abs(refined['normal']@n)-1)<1e-10
    result.update(nearby_surface_band_pollution_removed=True,all_exact_plane_points_retained=True,
        robust_plane_refinement_rigid_transform_control=True)
    return result
