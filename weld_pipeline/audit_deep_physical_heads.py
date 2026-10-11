"""Full-FIT diagnosis of frozen physical-event heads and decoding effects."""
import gc
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame,restore_probability_order
from weld_pipeline.continuous_center_modes import RESOLUTION_MM,center_modes,restore_event_order
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.continuous_vote_decoding import decode_vote_grids
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.neural_physical_event_sets import PhysicalEventNet
from weld_pipeline.neural_physical_event_votes import point_features,physical_votes
from weld_pipeline.train_forest_event_centers import load_cached

RUN=Path('weld_pipeline/runs/deep_physical_events_170_v3')
OUT=Path('weld_pipeline/reports/deep_physical_head_diagnosis_173_20261002')
FACTORS=('foreground','support','offsets')
VARIANTS={('+'.join(k for k,v in zip(FACTORS,flags) if v) or 'learned'):flags
          for flags in itertools.product((False,True),repeat=3)}


def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')


def compact_stat(values):
    values=np.concatenate(values) if values else np.empty(0)
    return dict(samples=len(values),median=float(np.median(values)) if len(values) else None,
                p95=float(np.quantile(values,.95)) if len(values) else None,
                within_3=float(np.mean(values<=3.)) if len(values) else None)


@torch.no_grad()
def main():
    if OUT.exists():raise ValueError('Preserve the frozen full-FIT diagnosis')
    complete=json.loads((RUN/'complete.json').read_text('utf8'))
    config=json.loads((RUN/'config.json').read_text('utf8'))
    variant=sorted(complete['variants'],key=lambda v:(-complete['variants'][v]['fit']['valid_children'],complete['variants'][v]['fit']['false_predictions'],v))[0]
    checkpoint=RUN/variant/'final.pt';frozen=sha256(checkpoint)
    if frozen!=complete['variants'][variant]['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    for path,digest in config['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Frozen physical model source changed')
    OUT.mkdir(parents=True);snap=OUT/'source_snapshot';snap.mkdir();shutil.copy2(__file__,snap/Path(__file__).name)
    dump(OUT/'protocol.json',dict(variant=variant,selected_for_diagnosis_only=True,
        selection='max FIT TP, min FIT FP, variant name',checkpoint_sha256=frozen,
        sources=config['train'],expected_truth_children=1221,source_sha256={__file__:sha256(__file__)},
        teacher_and_legacy_decoder_are_diagnostic_only_not_runtime_adoption=True,
        observed_held_or_formal_test_used=False,actual_trained_models=0,full_contract_pass=False))
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load(checkpoint,map_location=device);model=PhysicalEventNet().to(device);model.load_state_dict(saved['state_dict']);model.eval()
    mean=np.asarray(saved['mean'],np.float32);std=np.asarray(saved['std'],np.float32)
    score_sets={k:{} for k in [*VARIANTS,'grid_legacy','grid_continuous']}
    binary={k:dict(known=0,tp=0,fn=0,fp=0,tn=0) for k in FACTORS[:1]+('start_support','end_support','cut_support')}
    errors={k:[] for k in ('start','end','cut')};slopes={k:[] for k in errors}
    mode_counts={k:dict(learned=0,teacher=0) for k in errors};receipts={};began=time.perf_counter()
    for name in config['train']:
        data,near,known,piece=load_cached(name);sequences=piece['sequences']
        fg={v:[np.zeros(len(s['xyz']),np.float32) for s in sequences] for v in VARIANTS}
        events={v:[[[],[],[]] for s in sequences] for v in VARIANTS}
        grid=[np.zeros((len(s['xyz']),7),np.float32) for s in sequences]
        all_raw=[np.zeros((len(s['xyz']),7),np.float32) for s in sequences]
        for entry in data['entries']:
            index=entry['sequence']
            if entry['type']!='horizontal' or sequences[index].get('curve_kind')=='arc':continue
            seq,_,transform=canonical_frame(sequences[index]);size=len(seq['xyz']);step=float(seq['sample_step_mm'])
            raw=all_raw[index]
            for a in range(0,size,8192):
                b=min(size,a+8192);x=point_features(seq,data,entry,np.arange(a,b),saved['actual_xyz'])
                raw[a:b]=model(torch.from_numpy((x-mean)/std).to(device)).cpu().numpy()
            sigmoid=lambda x:1./(1.+np.exp(-np.clip(x.astype(float),-60.,60.)))
            learned_fg=sigmoid(raw[:,0]);learned_support=sigmoid(raw[:,1:4]);learned_offsets=raw[:,4:7].astype(float)
            a,b=entry['start'],entry['stop'];yy=np.asarray(near[a:b],float);kk=np.asarray(known[a:b])
            for role,key in enumerate(('foreground','start_support','end_support','cut_support')):
                valid=kk[:,role]>0;y=yy[:,role]>.5;p=(learned_fg if role==0 else learned_support[:,role-1])>=.5
                row=binary[key];row['known']+=int(valid.sum())
                for key2,mask in [('tp',y&p),('fn',y&~p),('fp',~y&p),('tn',~y&~p)]:row[key2]+=int((valid&mask).sum())
            positions=np.arange(size,dtype=float)*step
            for role,key in enumerate(('start','end','cut')):
                valid=kk[:,role+4]>0
                errors[key].append(np.abs(learned_offsets[:,role]-yy[:,role+4])[valid]*16.)
                truth_centers=positions+yy[:,role+4]*16.
                coherent=valid[:-1]&valid[1:]&(np.abs(np.diff(truth_centers))<=RESOLUTION_MM)
                slope_error=np.abs(np.diff(learned_offsets[:,role])*16./step+1.)
                slopes[key].append(slope_error[coherent])
            grid[index]=restore_probability_order(physical_votes(learned_fg,learned_support,learned_offsets,step),transform)
            for v,flags in VARIANTS.items():
                f=learned_fg.copy();s=learned_support.copy();o=learned_offsets.copy()
                if flags[0]:f[kk[:,0]>0]=yy[kk[:,0]>0,0]
                if flags[1]:s[kk[:,1:4]>0]=yy[:,1:4][kk[:,1:4]>0]
                if flags[2]:o[kk[:,4:7]>0]=yy[:,4:7][kk[:,4:7]>0]
                event=center_modes(s,o,step)
                fg[v][index],events[v][index]=restore_event_order(f,event,(size-1)*step,transform)
                if v in ('learned','foreground+support+offsets'):
                    for role,key in enumerate(('start','end','cut')):
                        mode_counts[key]['learned' if v=='learned' else 'teacher']+=len(event[role])
        fixed=json.loads((RUN/variant/'FIT'/(name+'.json')).read_text('utf8'))
        learned_predictions=decode_event_sets(sequences,fg['learned'],events['learned'])
        with np.load(RUN/variant/'FIT'/(name+'_foregrounds.npz'),allow_pickle=False) as archive:
            exact_fg=np.array_equal(np.concatenate(fg['learned']),archive['foregrounds'])
        if not exact_fg or learned_predictions!=fixed['predictions'] or events['learned']!=fixed['event_sets']:
            raise ValueError('Diagnostic learned baseline differs from frozen FIT output')
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        results={}
        for v in VARIANTS:
            predictions=learned_predictions if v=='learned' else decode_event_sets(sequences,fg[v],events[v])
            score=score_children(predictions,truth);score_sets[v][name]=score;results[v]=score
        for v,decoder in [('grid_legacy',lambda:decode_intervals(sequences,grid,threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)),
                          ('grid_continuous',lambda:decode_vote_grids(sequences,grid))]:
            score=score_children(decoder(),truth);score_sets[v][name]=score;results[v]=score
        np.savez_compressed(OUT/(name+'_raw_heads.npz'),raw=np.concatenate(all_raw),lengths=np.asarray([len(s['xyz']) for s in sequences]))
        dump(OUT/(name+'.json'),dict(baseline_foreground_events_and_children_exact=True,scores=results,
             diagnostic_teachers_and_legacy_grid_not_adopted=True,full_contract_pass=False))
        receipts[name]=sha256(OUT/(name+'.json'))
        print('DEEP_HEAD_CAUSAL',name,{v:(s['valid_children'],s['false_predictions']) for v,s in results.items()},flush=True)
        del data,near,known,piece,sequences,fg,events,grid,all_raw,learned_predictions;gc.collect()
    totals={v:aggregate(s) for v,s in score_sets.items()}
    if any(s['truth_count']!=1221 for s in totals.values()):raise ValueError('Full FIT child denominator changed')
    if sha256(checkpoint)!=frozen:raise ValueError('Frozen deep weights changed during diagnosis')
    dump(OUT/'complete.json',dict(protocol_sha256=sha256(OUT/'protocol.json'),checkpoint_sha256=frozen,
        variants=totals,sample_binary_metrics_not_child_acceptance=binary,
        sample_center_errors_mm_not_child_acceptance={k:compact_stat(v) for k,v in errors.items()},
        sample_offset_slope_error_from_minus_one={k:compact_stat(v) for k,v in slopes.items()},
        predicted_vs_teacher_event_mode_counts=mode_counts,piece_receipts=receipts,
        full_16_frozen_learned_outputs_exact=True,actual_trained_models=0,default_predictor_replaced=False,
        held_or_observed_read=False,elapsed_seconds=time.perf_counter()-began,full_contract_pass=False))
    print('DEEP_HEAD_CAUSAL_COMPLETE',{v:(s['valid_children'],s['false_predictions']) for v,s in totals.items()},flush=True)


if __name__=='__main__':main()
