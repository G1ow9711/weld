"""Fixed-budget single-case boundary encoding comparison against run193."""
from weld_pipeline.sjg_single_case_calibrated_round_1013 import main

if __name__=='__main__':
    main(default_prepared_run='run191', default_prediction_run='run196', default_anchored=True)
