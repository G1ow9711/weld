"""TRAIN21 finite-support diagnosis with observed ideal-plane residuals."""
from weld_pipeline.sjg_native_finite_pilot_983 import main
from weld_pipeline.sjg_ideal_plane_support_997 import extract,controls

if __name__=='__main__':
    main(extractor=extract,mesh_method='robust_ideal_plane_support_symmetric_2D_mesh',additional_controls=controls)
