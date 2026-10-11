"""Freeze both material-context endpoint arms and fixed learned interval/parent outputs before GT."""
import argparse
import sys
import traceback
import torch
import numpy as np
import shutil
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc
from weld_pipeline.sjg_weld_context_model_888 import MaterialSequenceModel
from weld_pipeline.sjg_weld_material_context_886 import WeldMaterialContext,CONTEXT_NAMES
from weld_pipeline.sjg_adjacency_eval_842 import ThirdSurfaceContext
from weld_pipeline.sjg_event_pairs_824 import decode_event_pairs
from weld_pipeline.sjg_free_sequence_823 import main as infer_sequence
from weld_pipeline.sjg_all_interval_data_844 import main as prepare_pairs
from weld_pipeline.sjg_evaluate_pairs_847 import main as infer_pairs,evaluate_frozen


class ZeroMaterialContext(ThirdSurfaceContext):
    """Matched control: absent material inputs remain zero as during training."""
    def __call__(self,profile,q):return np.c_[super().__call__(profile,q),np.zeros((len(q),8),np.float32)]

    def describe(self):return dict(super().describe(),material_features=CONTEXT_NAMES,
        material_inputs_zero=True,material_projection_verified_zero=True,material_identity_still_unknown=True)


class CompatibleWeldMaterialContext(WeldMaterialContext):
    def describe(self):
        result=super().describe();third=result.pop('third_surface');return dict(third,**result)


def main(reuse_control=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);arms=['control','material_context'];saved=sys.argv[:]
    assert read(OUT/'run89/receipt.json')['status']=='TWO_MATERIAL_CONTEXT_ARMS_TRAINED'
    write(dest/'registered_protocol.json',dict(stage='WELD_MATERIAL_CONTEXT_FREE_OUTPUT_COMPARISON',training_run='run89',
        arms=arms,original_G=496,optimizer_updates=0,fixed_pairing_checkpoint='run56',fixed_parent_link_checkpoint='run47',
        no_new_pair_training=True,no_threshold_tuning=True,all_arms_frozen_before_any_GT=True,
        control_material_inputs_zero_as_during_training=True,control_unused_material_recomputation_skipped=True,
        source_sha256=digest(__file__),candidate_generator_sha256=digest(ROOT/'weld_pipeline/sjg_all_interval_data_844.py'),
        pair_driver_sha256=digest(ROOT/'weld_pipeline/sjg_evaluate_pairs_847.py'),
        improvement_rule='development TP strictly increases and FP does not increase; all8 TP increases and FP does not increase versus matched control',
        full_contract_pass=False,start_utc=utc()))
    try:
        frozen=[]
        for arm in arms:
            folder=dest/arm;folder.mkdir()
            if arm=='control':
                with (OUT/'run89/control/last.pt').open('rb') as h:state=torch.load(h,map_location='cpu')
                assert torch.count_nonzero(state['model']['material.weight'])==0;del state
            if arm=='control' and reuse_control is not None:
                previous=OUT/reuse_control
                assert read(previous/'receipt.json')['status']=='FREE_SEQUENCE_PREDICTIONS_FROZEN_SCORE_PENDING'
                assert read(previous/'registered_protocol.json')['sequence_checkpoint_sha256']==digest(OUT/'run89/control/last.pt')
                shutil.copytree(previous,folder/'sequence');changes=[]
                for binding in read(previous/'all_predictions_frozen.json')['sources']:
                    assert digest(previous/binding['file'])==binding['sha256']
                    assert digest(folder/'sequence'/binding['file'])==binding['sha256']
                    fields=read(previous/f"source_{binding['source']:02d}_fields.json")['files']
                    for f in fields:assert digest(folder/'sequence'/f['file'])==f['sha256']
                    path=folder/'sequence'/f"source_{binding['source']:02d}_extra_context.json"
                    before=digest(path);context=read(path);third=context.pop('third_surface');write(path,dict(third,**context))
                    changes.append(dict(source=binding['source'],before_sha256=before,after_sha256=digest(path)))
                write(folder/'sequence/context_interface_recovery.json',dict(previous=str(previous.relative_to(OUT)),
                    only_metadata_events_moved_to_original_top_level=True,predictions_and_fields_unchanged=True,changes=changes))
                print('CONTROL_SEQUENCE_REUSED_WITH_CONTEXT_METADATA_FIX',len(changes),flush=True)
            else:
                sys.argv=[saved[0],'--run',str((folder/'sequence').relative_to(OUT)),'--checkpoint-run','run89/'+arm]
                infer_sequence(decode_event_pairs,score_outputs=False,model_factory=MaterialSequenceModel,
                    context_factory=ZeroMaterialContext if arm=='control' else CompatibleWeldMaterialContext)
            sys.argv=[saved[0],'--run',str((folder/'candidates').relative_to(OUT))]
            prepare_pairs(base=folder/'sequence',create_labels=False,expected_candidates=None)
            sys.argv=[saved[0],'--run',str((folder/'output').relative_to(OUT))]
            infer_pairs(data=folder/'candidates',score_outputs=False)
            frozen.append(dict(arm=arm,sequence_sha256=digest(folder/'sequence/all_predictions_frozen.json'),
                candidates_sha256=digest(folder/'candidates/all_inputs_frozen.json'),outputs_sha256=digest(folder/'output/all_outputs_frozen_before_GT.json')))
        write(dest/'all_arms_frozen_before_GT.json',dict(arms=frozen,all_before_GT=True))
        results={arm:evaluate_frozen(dest/arm/'output','run56') for arm in arms}
        write(dest/'receipt.json',dict(status='WELD_MATERIAL_CONTEXT_COMPARISON_EVALUATED',arms=results,original_G=496,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise
    finally:sys.argv=saved


if __name__=='__main__':main()
