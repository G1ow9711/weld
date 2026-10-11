"""Loss masking and free typed decoder controls before the training round."""
import argparse
from weld_pipeline.sjg_local_runtime_904 import OUT,write,utc
from weld_pipeline.sjg_local_event_train_930 import controls as loss_controls
from weld_pipeline.sjg_local_event_infer_931 import controls as decoder_controls
from weld_pipeline.sjg_local_event_score_932 import controls as score_controls
from weld_pipeline.sjg_local_graph_data_916 import neighborhood_controls
from weld_pipeline.sjg_material_curve_model_953 import controls as material_controls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    result=dict(loss=loss_controls(),decoder=decoder_controls(),scoring=score_controls(),neighborhood=neighborhood_controls(),material_model=material_controls())
    write(dest/'controls.json',result)
    write(dest/'receipt.json',dict(status='LOCAL_EVENT_CONTROLS_PASSED',controls=result,optimizer_updates=0,end_utc=utc()))
    print('LOCAL_EVENT_LOSS_DECODER_SCORE_CONTROLS_PASSED',flush=True)

if __name__=='__main__':main()
