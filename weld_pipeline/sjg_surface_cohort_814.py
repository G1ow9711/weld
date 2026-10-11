"""Freeze the two-TRAIN-developed extractor, then run the original eight cases."""
import argparse
from weld_pipeline.sjg_prepare_776 import TRAIN,VALID,ARC
from weld_pipeline.sjg_surface_pilot_811 import run
from weld_pipeline.sjg_global_membership_813 import extract


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    run(args,TRAIN+VALID+ARC,extractor=extract,extra_protocol=dict(name='frozen_learned_global_membership',
        developed_only_on_train_sources=[21,4],development_result='run32',
        maximum_normal_angle_deg=20.,minimum_boundary_probability=.5,
        shared_boundary=True,point_plane_distance_mm=.2,unconditional_global_fill=False,
        validation_threshold_tuning=False,all_original_pilot_points_retained=True))


if __name__=='__main__':main()
