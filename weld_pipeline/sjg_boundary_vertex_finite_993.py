"""Retain predicted boundary vertices through finite-surface tessellation."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_local_sampling_mesh_991 import extract_boundary_vertices,boundary_vertex_controls

if __name__=='__main__':
    main(extractor=extract_boundary_vertices,mesh_method='local_2D_sampling_boundary_vertices_shared',additional_controls=boundary_vertex_controls)
