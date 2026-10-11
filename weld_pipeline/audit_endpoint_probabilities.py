"""Reference-aligned heatmap diagnostics after a checkpoint is frozen."""
import argparse
import json
from pathlib import Path
import numpy as np
from weld_pipeline.train_typed_pilot import load
from weld_pipeline.model_registry_extended import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--data', default='weld_pipeline/data/typed_pilot_v5')
    ap.add_argument('--group', required=True)
    args = ap.parse_args()
    root = Path(args.run)
    state = load(root/'best.pt')
    piece = load(Path(args.data)/(args.group+'.pt'))
    model = model_from_checkpoint(state).cuda().eval()
    probabilities = infer_probabilities(model, piece['sequences'])
    rows = []
    for wi, weld in enumerate(piece['truth']):
        mapping = piece['record']['mapping']['weld_to_sequence']
        si = mapping.get(wi, mapping.get(str(wi)))
        if si is None:continue
        seq, prob = piece['sequences'][si], probabilities[si]
        pts = np.array([weld['segments'][0]['points'][0], weld['segments'][-1]['points'][-1]])
        positions = np.sort((pts-seq['xyz'][0]) @ seq['direction']/seq['sample_step_mm'])
        row = {'id':weld['id'], 'name':weld['name'], 'type':weld['type'], 'sequence':si,
               'segments':len(weld['segments']), 'endpoints':pts.tolist()}
        for index, channel in enumerate([1, 2]):
            i = int(round(positions[index]))
            a, b = max(0, i-1), min(len(prob), i+2)
            row['start_heat' if index == 0 else 'end_heat'] = float(prob[a:b, channel].max())
        a, b = max(0, int(np.ceil(positions[0]))), min(len(prob), int(np.floor(positions[1]))+1)
        if b > a:
            row['mask_mean'] = float(prob[a:b, 0].mean())
            row['work_range_mean'] = float(prob[a:b, 5].mean())
            row['dual_robot_mean'] = float(prob[a:b, 6].mean())
        rows.append(row)
    dest = root/(args.group+'_endpoint_heat_audit.json')
    dest.write_text(json.dumps({'epoch':state['epoch'], 'scope':'Reference-aligned diagnostics, not inference', 'rows':rows}, indent=2), encoding='utf-8')
    for typ in ['horizontal', 'vertical']:
        subset = [r for r in rows if r['type'] == typ]
        if subset:
            print(typ, {'truth':len(subset), 'start_heat_pass':sum(r['start_heat']>=.5 for r in subset),
                        'end_heat_pass':sum(r['end_heat']>=.5 for r in subset),
                        'mask_mean_pass':sum(r.get('mask_mean',0)>=.5 for r in subset)}, flush=True)
    for r in rows[:8]:print({k:v for k,v in r.items() if k not in ['id','endpoints']}, flush=True)


if __name__ == '__main__':main()
