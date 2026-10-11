"""Inspect supervised sequences without altering predictions or acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.constrained_decoder import refined_peaks
from weld_pipeline.model_registry_context import model_from_checkpoint


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--outdir',required=True)
    ap.add_argument('--numbers',nargs='+',required=True);a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    with Path(a.checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    for source,h in state['source_hashes'].items():
        if sha256(source)!=h:raise ValueError('Source changed: '+source)
    folder=Path('weld_pipeline/data/mixed_surface_v1')/a.group;piece=load_piece(folder)
    if sha256(folder/'manifest.json')!=state['data_hashes'][a.group]:raise ValueError('Cache changed')
    model=model_from_checkpoint(state).cuda().eval();prob=infer_probabilities(model,piece['sequences'])
    np.savez_compressed(out/'probabilities.npz',**{str(i):p for i,p in enumerate(prob)})
    rows=[];lines=[w for w in piece['truth'] if all(s['kind']=='line' for s in w['segments'])]
    line_ids=[i for i,s in enumerate(piece['sequences']) if s.get('curve_kind')!='arc']
    mapping=piece['record']['mapping']['line_mapping']['weld_to_sequence']
    for wi,w in enumerate(lines):
        number=w['metadata']['WeldNumber (f/v + Name number)']
        if number not in a.numbers:continue
        si=line_ids[mapping[str(wi)]];seq=piece['sequences'][si];y=piece['targets'][si];p=prob[si];step=seq['sample_step_mm']
        peaks=lambda arr:{str(c):[[round(float(t*step),4),round(h,5)] for t,h in refined_peaks(arr[:,c],step,.5)] for c in [1,2,3]}
        dec=decode_curved([seq],[p]);target=decode_curved([seq],[y])
        duplicates=[]
        for i,other in enumerate(piece['sequences']):
            if i==si or other.get('curve_kind')=='arc' or other['type']!=seq['type']:continue
            if abs(np.dot(other['direction'],seq['direction']))<.9999:continue
            error=float(np.linalg.norm(np.cross(other['xyz'][0]-seq['xyz'][0],seq['direction'])))
            if error<.25:duplicates.append({'sequence':i,'line_distance_mm':error,'foreground_points':int(piece['targets'][i][:,0].sum())})
        rows.append({'weld_number':number,'sequence':si,'normal1':seq['features'][len(p)//2,60:63].tolist(),
                     'normal2':seq['features'][len(p)//2,63:66].tolist(),'z_bounds_mm':[float(seq['xyz'][:,2].min()),float(seq['xyz'][:,2].max())],
                     'target_peaks_mm':peaks(y),'predicted_peaks_mm':peaks(p),
                     'foreground_mean_probability':float(p[y[:,0]>.5,0].mean()),
                     'predictions':dec,'target_decoded':target,'collinear_sequences':duplicates})
    report={'group':a.group,'checkpoint_sha256':sha256(a.checkpoint),'cache_sha256':sha256(folder/'manifest.json'),
            'source_sha256':sha256(Path(__file__)),'scope':'Training fit diagnostic; no thresholds or labels changed.','rows':rows}
    (out/'sequence_fit_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:print(json.dumps({k:r[k] for k in ['weld_number','sequence','target_peaks_mm','predicted_peaks_mm','foreground_mean_probability','collinear_sequences']}),flush=True)


if __name__=='__main__':main()
