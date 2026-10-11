"""Recover streamed inference by reclaiming only this worker's idle pages."""
from weld_pipeline.sjg_local_stream_features_906 import main

if __name__=='__main__':main(reclaim_working_set=True)
