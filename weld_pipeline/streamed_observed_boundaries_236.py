"""Current-XYZ complete finite faces with all material candidates on disk."""
from dataclasses import asdict
import numpy as np

from .finite_plate_binding_v1_204 import CandidateConfig
from .fresh_observed_boundaries_234 import FreshObservedBoundaries,DETECTOR_SCOPE
from .primitive_incidence_209 import array_sha
from .runtime_primitive_provider_adapter_212 import ObservationBinding,ProviderError
from .streamed_material_candidates_236 import StreamedMaterialCandidates

VERSION='FRESH_XYZ_ALL_FINITE_FACES_COMPLETE_STREAMED_MATERIAL_CANDIDATES_236_v1'


class StreamedObservedBoundaries(FreshObservedBoundaries):
    def __init__(self,xyz_mm,binding,config=CandidateConfig(),*,material_directory,progress=None):
        if not isinstance(binding,ObservationBinding) or binding.kind!='runtime_pcd':
            raise ProviderError('Current runtime PCD observation binding required')
        xyz=np.asarray(xyz_mm)
        if xyz.dtype!=np.float64 or xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
            raise ProviderError('Complete finite float64 original millimeter XYZ required')
        self.xyz,self.binding=xyz,binding
        self.before_xyz_sha256=array_sha(xyz)
        self.candidates=StreamedMaterialCandidates.from_raw_xyz(xyz,config,
            material_directory=material_directory,progress=progress)
        if array_sha(xyz)!=self.before_xyz_sha256:
            raise ProviderError('Discovery changed original raw XYZ/order')
        self.source_receipt=dict(version=VERSION,current_observation_sha256=binding.observation_sha256,
            original_xyz_array_sha256=self.before_xyz_sha256,original_points=len(xyz),
            source_names_or_coordinates_used_as_identity=False,cached_geometry_or_features_read=False,
            actual_fresh_plane_and_face_discovery=True,detector_scope=DETECTOR_SCOPE,
            discovery_config=asdict(config),finite_faces=len(self.candidates.faces),
            material_pair_candidates=len(self.candidates.plates),
            complete_material_pair_stores=self.candidates.material_store_bindings,
            original_raw_row_order_and_values_preserved=True,GT_STEP_Excel_used=False,
            material_identity_verified=False,weld_identity_or_type_verified=False,
            no_short_boundary_edge_deleted=True,all_material_candidates_preserved=True,full_contract_pass=False)
