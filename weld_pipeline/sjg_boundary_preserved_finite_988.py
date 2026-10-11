"""Same anisotropic reconstruction with per-cell boundary extrema preserved."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_anisotropic_mesh_986 import extract_preserve_boundary,boundary_controls

if __name__=='__main__':
    main(extractor=extract_preserve_boundary,mesh_method='anisotropic_with_cell_extrema',additional_controls=boundary_controls)
