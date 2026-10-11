"""Same TRAIN21 weights/planes, with observed anisotropic sampling geometry."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_anisotropic_mesh_986 import extract,controls

if __name__=='__main__':
    main(extractor=extract,mesh_method='observed_periodic_anisotropic',additional_controls=controls)
