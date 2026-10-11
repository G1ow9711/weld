"""Infer sampling directions independently of the plane PCA axes."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_anisotropic_mesh_986 import extract_oriented_sampling,oriented_controls

if __name__=='__main__':
    main(extractor=extract_oriented_sampling,mesh_method='oriented_sampling_extrema_shared',additional_controls=oriented_controls)
