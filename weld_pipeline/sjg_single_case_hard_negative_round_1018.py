"""Fixed-budget single-TRAIN hard-negative sampling correction."""
from weld_pipeline.sjg_single_case_calibrated_round_1013 import main

if __name__=='__main__':
    main(default_prediction_run='run198', default_anchored=True, default_hard_negative_run='run196')
