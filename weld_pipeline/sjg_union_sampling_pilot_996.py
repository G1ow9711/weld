"""Fixed TRAIN21 pilot of symmetric local sampling support."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_union_sampling_mesh_995 import extract,controls

if __name__=='__main__':
    main(extractor=extract,mesh_method='symmetric_local_2D_sampling_union_boundary_vertices_shared',additional_controls=controls)
