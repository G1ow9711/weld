"""Choose one shared interval decoder configuration from TRAIN groups only."""
import argparse,gc,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece as load_curved_piece,STORAGE_FORMAT
from weld_pipeline.mapped_cache import load_piece as load_mapped_piece
from weld_pipeline.model_registry_crossing import model_from_checkpoint
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.interval_decoder import decode_intervals
from weld_pipeline.fast_counts import score_counts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--outdir',required=True)
    ap.add_argument('--data',default='weld_pipeline/data/mixed_surface_v1')
    a=ap.parse_args();root=Path(a.data);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    if (out/'decoder_config.json').exists():raise ValueError('Frozen calibration exists; use a new output directory')
    split=json.loads((root/'split.json').read_text(encoding='utf-8'));torch.set_num_threads(4)
    with Path(a.checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    historical_sources=verify_cached_inference_sources(state,a.checkpoint)
    load_piece=load_curved_piece if split.get('storage_format')==STORAGE_FORMAT else load_mapped_piece
    model=model_from_checkpoint(state).cuda().eval()
    configurations=[{'threshold':.5,'weld_penalty':penalty,'mask_weight':weight} for penalty in [0.,2.,4.,6.] for weight in [.25,.5,1.]]
    scores=[{} for _ in configurations];start=time.perf_counter();cache_hashes={}
    training_groups=split.get('augmentation_policy',{}).get('original_train',split['train'])
    if not set(training_groups)<=set(split['train']) or set(training_groups)&set(split['validation']):
        raise ValueError('Calibration groups violate frozen TRAIN membership')
    for name in training_groups:
        folder=(root/split.get('piece_roots',{}).get(name,name)).resolve();wanted=sha256(folder/'manifest.json')
        if state['data_hashes'][name]!=wanted:raise ValueError('Training cache changed')
        piece=load_piece(folder);cache_hashes[name]=wanted
        probabilities=infer_probabilities(model,piece['sequences'],halo=getattr(model,'inference_halo',256))
        for index,config in enumerate(configurations):
            pred=decode_intervals(piece['sequences'],probabilities,**config)
            scores[index][name]=score_counts(pred,piece['truth'])
        print('CALIBRATED_TRAIN',name,flush=True)
        (out/'partial_scores.json').write_text(json.dumps({'configurations':configurations,'scores':scores},indent=2),encoding='utf-8')
        del piece,probabilities,pred;gc.collect()
    quality=[]
    for rows in scores:
        passed=[r['valid_welds']/r['truth_count']>.95 and r['false_or_duplicate_predictions']/r['truth_count']<=.05 for r in rows.values()]
        macro=float(np.mean([2*r['valid_welds']/max(1,r['truth_count']+r['prediction_count']) for r in rows.values()]))
        quality.append((sum(passed),macro,-sum(r['false_or_duplicate_predictions'] for r in rows.values())))
    chosen=max(range(len(configurations)),key=lambda i:quality[i])
    source_names=['interval_decoder.py','constrained_decoder.py','curved_sequences.py']
    sources={str(Path('weld_pipeline')/name):sha256(Path('weld_pipeline')/name) for name in source_names}
    report={'scope':'TRAIN calibration only; validation labels have not been loaded.','checkpoint_sha256':sha256(a.checkpoint),
            'training_groups':training_groups,'augmentation_entries_excluded_from_calibration':[n for n in split['train'] if n not in training_groups],
            'data_hashes':cache_hashes,'configurations':configurations,'scores':scores,'quality':quality,
            'chosen_index':chosen,'elapsed_seconds':time.perf_counter()-start,'selection':'Training group pass count, then macro F1, then fewer false predictions.',
            'data_root':str(root),'split_sha256':sha256(root/'split.json'),'historical_entrypoints':historical_sources,
            'sources':{**sources,str(Path(__file__)):sha256(Path(__file__)),
                       'weld_pipeline/checkpoint_sources.py':sha256('weld_pipeline/checkpoint_sources.py')}}
    receipt=out/'calibration_receipt.json';receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    configuration={'version':'shared_interval_decoder_v1','checkpoint_sha256':sha256(a.checkpoint),'parameters':configurations[chosen],
                   'sources':sources,'training_calibration_receipt_sha256':sha256(receipt)}
    (out/'decoder_config.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    snapshots=out/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for name in report['sources']:shutil.copy2(name,snapshots/Path(name).name)
    print('SELECTED',json.dumps({'parameters':configurations[chosen],'quality':quality[chosen],'TD2':scores[chosen].get('TD2-LB8A')}),flush=True)


if __name__=='__main__':main()
