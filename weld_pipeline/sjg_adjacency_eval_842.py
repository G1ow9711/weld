"""Rebuild finite geometry context from raw-derived surfaces for free inference."""
import argparse
import sys
import traceback
import numpy as np
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_train_776 import utc
from weld_pipeline.sjg_third_surface_838 import finite_events,adjacency_features
from weld_pipeline.sjg_adjacency_model_840 import AdjacencySequenceModel
from weld_pipeline.sjg_event_pairs_824 import decode_event_pairs
from weld_pipeline import sjg_free_sequence_823 as free_sequence
from weld_pipeline.sjg_endpoint_compare_837 import freeze_parent,evaluate_parent


class ThirdSurfaceContext:
    def __init__(self,xyz,faces,profiles,selected):
        archives={};bindings={}
        def loader(face):
            path=ROOT/face['support_file'];key=str(path)
            if key not in archives:
                assert digest(path)==face['support_sha256'];archives[key]=np.load(path);bindings[key]=face['support_sha256']
            z=archives[key];return z[face['point_ids_key']],z[face['triangles_key']]
        try:self.events=finite_events(xyz,faces,profiles,selected,loader,lambda *a:print('RUNTIME_CONTEXT',*a,flush=True))
        finally:
            for archive in archives.values():archive.close()
        self.bindings=bindings

    def __call__(self,profile,q):return adjacency_features(q,self.events[profile])

    def describe(self):return dict(events=self.events,support_bindings=self.bindings,
        no_source_ID_lookup=True,GT_and_STEP_reads=0,rebuilt_from_input_geometry=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);saved=sys.argv[:]
    write(dest/'registered_protocol.json',dict(stage='FINITE_THIRD_SURFACE_FREE_OUTPUT_AND_FROZEN_PARENT',
        training_run='run52',actual_optimizer_updates=0,original_G=496,
        compare_to='run50/control_parents',all_predictions_before_GT=True,
        runtime_context_recomputed_not_read_from_run51=True,no_fixed_cutback=True,
        source_sha256=digest(__file__),feature_sha256=digest(ROOT/'weld_pipeline/sjg_third_surface_838.py'),
        free_driver_sha256=digest(ROOT/'weld_pipeline/sjg_free_sequence_823.py'),
        model_sha256=digest(ROOT/'weld_pipeline/sjg_adjacency_model_840.py'),
        parent_driver_sha256=digest(ROOT/'weld_pipeline/sjg_endpoint_compare_837.py'),
        checkpoint_sha256=digest(OUT/'run52/last.pt'),full_contract_pass=False,start_utc=utc()))
    try:
        sys.argv=[saved[0],'--run',str((dest/'free').relative_to(OUT)),'--checkpoint-run','run52']
        free_sequence.main(decode_event_pairs,score_outputs=False,model_factory=AdjacencySequenceModel,context_factory=ThirdSurfaceContext)
        freeze_parent(dest/'free',dest/'parents')
        write(dest/'all_stages_frozen_before_GT.json',dict(free_sha256=digest(dest/'free/all_predictions_frozen.json'),
            parents_sha256=digest(dest/'parents/all_predictions_frozen.json'),all_before_evaluation_GT=True))
        free_rows=free_sequence.evaluate_outputs(dest/'free');parent_rows=evaluate_parent(dest/'parents')
        write(dest/'receipt.json',dict(status='FINITE_ADJACENCY_FREE_AND_PARENT_EVALUATED',free=free_rows,parents=parent_rows,
            original_G=496,actual_optimizer_updates=0,full_contract_pass=False,E16_and_material_unresolved=True,
            curves_missing=True,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise
    finally:sys.argv=saved


if __name__=='__main__':main()
