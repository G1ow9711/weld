"""Guarded raw-PCD replay of both frozen shared physical-event deep models."""
import builtins
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.forest_event_centers import build_inputs
from weld_pipeline.neural_physical_event_sets import PhysicalEventNet,infer_event_sets
from weld_pipeline.neural_physical_event_votes import extra_features
from weld_pipeline.train_forest_event_centers import load_cached

RUN=Path('weld_pipeline/runs/deep_physical_events_170_v3')
OUT=Path('weld_pipeline/reports/deep_physical_events_runtime_172_20261002')
CONTEXT=Path('weld_pipeline/data/joint_workpiece_context_pilot154')


def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')


def main():
    if OUT.exists():raise ValueError('Preserve the fixed deep raw runtime proof')
    complete=json.loads((RUN/'complete.json').read_text('utf8'))
    config=json.loads((RUN/'config.json').read_text('utf8'))
    for p,digest in config['source_sha256'].items():
        if sha256(p)!=digest:raise ValueError('Frozen deep model source changed: '+p)
    OUT.mkdir(parents=True);torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');results={}
    for variant,receipt in complete['variants'].items():
        checkpoint=RUN/variant/'final.pt';frozen=sha256(checkpoint)
        if frozen!=receipt['checkpoint_sha256']:raise ValueError('Frozen deep checkpoint changed')
        saved=torch.load(checkpoint,map_location=device)
        model=PhysicalEventNet().to(device);model.load_state_dict(saved['state_dict']);model.eval()
        results[variant]={}
        for name in ('207-FR79M','203-TT1A'):
            manifest=json.loads((CONTEXT/name/'manifest.json').read_text('utf8'))
            pcd=Path(manifest['input']).resolve();original=sha256(pcd)
            if original!=manifest['input_sha256']:raise ValueError('Original raw deep input changed')
            roots=[Path(p).resolve() for p in ['DataSet','weld_pipeline/data','weld_pipeline/reports']]
            openers=builtins.open,io.open,Path.open;reads=[];controls=[]
            def guard(opener):
                def checked(path,mode='r',*args,**kwargs):
                    if isinstance(path,(str,bytes,Path)) and ('r' in str(mode) or '+' in str(mode)):
                        file=Path(path).resolve()
                        if file==pcd:reads.append(str(file))
                        elif any(file.is_relative_to(root) for root in roots) or file.suffix.lower() in ('.stp','.step','.xls','.xlsx') or file.name.endswith('_welds.json'):
                            raise RuntimeError('Forbidden shared deep runtime reference/cache: '+str(file))
                    return opener(path,mode,*args,**kwargs)
                return checked
            began=time.perf_counter();builtins.open,io.open,Path.open=tuple(guard(opener) for opener in openers)
            try:
                for opener in (builtins.open,io.open,Path.open):
                    try:opener(Path('weld_pipeline/data/runtime_deep_physical_control.step'),'r')
                    except RuntimeError:controls.append(True)
                    else:raise ValueError('Deep raw runtime read guard failed')
                sequences,data=build_inputs(read_xyz_pcd(pcd)*1000.)
                foregrounds,event_sets=infer_event_sets(model,data,sequences,saved['mean'],saved['std'],saved['actual_xyz'])
                predictions=decode_event_sets(sequences,foregrounds,event_sets)
            finally:builtins.open,io.open,Path.open=openers
            lengths=np.asarray([len(p) for p in foregrounds]);fresh=np.concatenate(foregrounds)
            with np.load(RUN/variant/'FIT'/(name+'_foregrounds.npz'),allow_pickle=False) as archive:
                same_grid=np.array_equal(lengths,archive['lengths'])
                error=float(np.abs(fresh-archive['foregrounds']).max()) if same_grid else None
            expected=json.loads((RUN/variant/'FIT'/(name+'.json')).read_text('utf8'))
            cached,_,_,piece=load_cached(name)
            components={}
            for key in ('local','point_context','nodes'):
                same_shape=data[key].shape==cached[key].shape
                components[key]=dict(shape_exact=same_shape,values_exact=same_shape and np.array_equal(data[key],cached[key]),
                    max_error=float(np.abs(data[key]-cached[key]).max()) if same_shape else None)
            all_extra_exact=True
            for e in data['entries']:
                if e['type']!='horizontal':continue
                raw_seq,_,raw_transform=canonical_frame(sequences[e['sequence']])
                cache_seq,_,cache_transform=canonical_frame(piece['sequences'][e['sequence']])
                ids=np.arange(len(raw_seq['xyz']))
                if raw_transform!=cache_transform or not np.array_equal(extra_features(raw_seq,ids,saved['actual_xyz']),extra_features(cache_seq,ids,saved['actual_xyz'])):
                    all_extra_exact=False;break
            events_exact=event_sets==expected['event_sets'];predictions_exact=predictions==expected['predictions']
            passed=bool(same_grid and error==0. and events_exact and predictions_exact and all_extra_exact
                and all(c['values_exact'] for c in components.values()) and len(controls)==3 and reads
                and sha256(pcd)==original and sha256(checkpoint)==frozen)
            result=dict(checkpoint_sha256=frozen,input_sha256=original,foreground_grid_exact=same_grid,
                foreground_max_error=error,continuous_event_sets_exact=events_exact,predictions_exact=predictions_exact,
                feature_components=components,physical_position_and_optional_actual_XYZ_exact=all_extra_exact,
                negative_controls=len(controls),PCD_reads=len(reads),runtime_STEP_used=False,runtime_labels_used=False,
                runtime_preprocessed_cache_used=False,all_checks_passed=passed,elapsed_seconds=time.perf_counter()-began,
                raw_runtime_proof_is_not_geometry_or_attribute_acceptance=True,full_contract_pass=False)
            dump(OUT/(variant+'_'+name+'.json'),result);results[variant][name]=result
            print('DEEP_PHYSICAL_RAW_RUNTIME',variant,name,passed,'FG_ERROR',error,'EVENTS_EXACT',events_exact,'PREDICTIONS_EXACT',predictions_exact,flush=True)
            if not passed:raise ValueError('Pure raw deep output differs from fixed FIT receipt')
            del cached,piece,data,sequences,foregrounds,event_sets,predictions,fresh
    dump(OUT/'complete.json',dict(variants=results,all_checks_passed=True,
        trained_models=len(results),raw_replay_count=sum(len(v) for v in results.values()),
        source_sha256={__file__:sha256(__file__)},frozen_training_complete_sha256=sha256(RUN/'complete.json'),
        default_predictor_replaced=False,full_contract_pass=False))


if __name__=='__main__':main()
