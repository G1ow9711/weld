"""Finite-surface reconstruction with shared observed junction memberships."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_anisotropic_mesh_986 import extract_shared_boundary,shared_controls

if __name__=='__main__':
    main(extractor=extract_shared_boundary,mesh_method='anisotropic_extrema_shared_membership',additional_controls=shared_controls)
