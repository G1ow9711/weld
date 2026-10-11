"""Associate frozen finite height evidence with actual input-only stations.

All compatible physical hypotheses remain. Error values are geometric estimates,
not certified sensor bounds or contract results. No target value is consumed.
"""
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.finite_face_pair_schedule_755 import TwoPlaneRows

VERSION='INPUT_STATION_OPPOSED_FINITE_HEIGHT_ASSOCIATION_V1'


class FiniteHeightAssociation:
    def __init__(self,xyz,groups,heights,payload_folder):
        self.xyz=xyz;self.groups=groups;self.heights=heights
        self.cache=TwoPlaneRows(payload_folder);self.last_group=None;self.tree=None

    def query(self,point,axes):
        point=np.asarray(point,np.float64);axes=np.asarray(axes,np.float64).reshape(2,3)
        if not np.isfinite(point).all() or not np.isfinite(axes).all():raise ValueError('Nonfinite physical query')
        norms=np.linalg.norm(axes,axis=1)
        if np.any(norms<1e-8):return []
        axes=axes/norms[:,None]
        if abs(float(axes[0]@axes[1]))>.15:return []
        upright_axes=[axes[i] for i in (0,1) if abs(axes[i,2])<=.15 and abs(axes[1-i,2])>=.95]
        if not upright_axes:return []
        hypotheses=[]
        for hi,height in enumerate(self.heights):
            gi=height['upright_alias_group'];face=self.groups[gi]['representative'];normal=np.asarray(face['normal'])
            if min(np.linalg.norm(np.cross(normal,a)) for a in upright_axes)>.15:continue
            residual=abs(float(point@normal-face['plane_offset_mm']))
            if residual>.15:continue
            coordinate=float(point@np.asarray(height['section_axis']))
            lo,z=height['finite_section_interval_mm']
            slack=64*np.finfo(np.float64).eps*max(1.,abs(coordinate),float(np.abs(point).max()))
            if not lo-slack<=coordinate<=z+slack:continue
            bottom=height['bottom'];top=height['top']
            low=bottom['z_slope']*coordinate+bottom['z_intercept_mm']
            high=top['z_slope']*coordinate+top['z_intercept_mm']
            if not low-.15<=point[2]<=high+.15:continue
            if self.last_group!=gi:
                self.tree=None
                rows=self.cache.pin(face)
                self.tree=cKDTree(np.asarray(self.xyz[rows],np.float64));self.last_group=gi
            nearest=float(self.tree.query(point,k=1)[0]);spacing=face['coverage']['original_resolution_mm']
            if nearest>1.5*spacing+.15:continue
            value=float(height['height_linear_slope']*coordinate+height['height_linear_intercept_mm'])
            # Stored features have float32 world XYZ/10000. Bound only that
            # representation rounding contribution; it is not sensor accuracy.
            position_rounding=np.finfo(np.float32).eps*max(1.,float(np.linalg.norm(point)))
            estimated=float(height['estimated_geometric_error_mm']+abs(height['height_linear_slope'])*position_rounding)
            valid=estimated<=3.
            features=np.array([value/1000.,spacing/10.,face['fit_residual_max_mm'],face['original_row_count']/100000.,
                face['coverage']['material_cells']/100000.,face['coverage']['raster_inner_regions'],abs(normal[2]),float(valid),
                1.,0.,0.,0.,1.,0.,0.,0.],np.float32)
            key=hashlib.sha256(repr((face['original_rows_sha256'],height['bottom'],height['top'],point.tolist())).encode()).hexdigest()
            hypotheses.append(dict(physical_value_mm=value,field_valid=valid,uncertainty_mm=estimated,features=features.tolist(),
                boundary_kind='ASSOCIATED_OPPOSED_FINITE_DIRECT_EDGES',order_status='ROLE_UPRIGHT_BASE_GEOMETRIC',sign_status='UNRESOLVED',
                physical_source_key=key,support_interval_mm=[max(0.,value-estimated),value+estimated],
                known_role_constraints_consistent=True,physical_plate_identity_confirmed=False,
                estimated_sampling_uncertainty_not_certified_error_bound=True,
                estimated_error_excludes_sensor_error_and_wrong_face_association=True,
                certified_boundary_error_bound_mm=None,
                evidence_height_index=hi,evidence_upright_group=gi,
                query_station_mm=point.tolist(),query_plane_residual_mm=residual,query_nearest_material_mm=nearest,
                observed_sampling_spacing_mm=spacing,all_compatible_physical_candidates_retained=True))
        return hypotheses

    def raw_record(self,features,whole_path):
        samples=np.asarray(features).reshape(17,84) if whole_path else np.asarray(features).reshape(1,84)
        result=dict(fields=[[] for _ in range(16)],unresolved_fields_explicit=True,
            input_only_current_XYZ_finite_height_association=True,whole_path_plate_identity_confirmed=False)
        for field,index in ((6,0),(7,-1)):
            row=samples[index]
            result['fields'][field]=self.query(row[67:70].astype(np.float64)*10000.,row[60:66])
        return result
