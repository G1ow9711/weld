"""Local sampling neighborhoods handle several strip directions on one face."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_local_sampling_mesh_991 import extract,controls

if __name__=='__main__':
    main(extractor=extract,mesh_method='local_2D_sampling_extrema_shared',additional_controls=controls)
