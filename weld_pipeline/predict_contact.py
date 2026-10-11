"""Experimental inference entry: raw PCD + checkpoint, no reference inputs."""
import argparse,json,time
from pathlib import Path
import torch
from training.dataio import read_xyz_pcd
from weld_pipeline.contact_sequences import make_sequences,FEATURE_VERSION
from weld_pipeline.contact_model import ContactNet,decode
from weld_pipeline.export import save_csv,save_all


def predict(pcd,checkpoint,outdir):
    torch.set_num_threads(4)
    with Path(checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    if state['feature_version']!=FEATURE_VERSION:raise ValueError('Checkpoint/feature version mismatch')
    model=ContactNet(state['features']).cuda().eval();model.load_state_dict(state['model'])
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    torch.cuda.synchronize();start=time.perf_counter()
    xyz=read_xyz_pcd(pcd)*1000;sequences,meta=make_sequences(xyz)
    with torch.no_grad():
        probabilities=[model(torch.from_numpy(s['features'].T.copy()).cuda().unsqueeze(0)).sigmoid()[0].T.cpu().numpy() for s in sequences]
    predictions=decode(sequences,probabilities)
    save_csv(predictions,out/'predicted_welds.csv');torch.cuda.synchronize();seconds=time.perf_counter()-start
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    save_all(predictions,out)
    receipt={'input_pcd':str(pcd),'checkpoint':str(checkpoint),'welds':len(predictions),'raw_pcd_to_csv_seconds':seconds,
        'model_load_timed':False,'excel_and_ply_timed':False,'repetitions':1,
        'reference_inputs_used':False,'geometry':meta,'scope':state['scope'],'acceptance':'not evaluated by inference'}
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pcd',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--outdir',required=True)
    a=p.parse_args();predict(a.pcd,a.checkpoint,a.outdir)
