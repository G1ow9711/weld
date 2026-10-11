"""Fresh PCD regeneration under a reference/cache read guard; compare after inference."""
import builtins
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.material_support_ablation import side_sorted
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.signed_material_support import SignedMaterialSupport, frame_views, predict_equivalent_views

OUT=Path('weld_pipeline/reports/material_support_raw_148_20261001')
DATA=Path('weld_pipeline/data/material_support_v1')
RUN=Path('weld_pipeline/runs/material_support_crossfit_seed148')
NAMES=('207-FR79M','203-TT1A')


def main():
    if OUT.exists():raise ValueError('Preserve raw material feature verification')
    config=json.loads((DATA/'config.json').read_text('utf8'))
    for source,digest in config['sources'].items():
        if sha256(source)!=digest:raise ValueError('Material source changed')
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    model_paths={target:RUN/('full_fit_seed148_signed_'+target+'.joblib') for target in ('material','business')}
    for path in model_paths.values():
        if not path.exists():raise ValueError('Finished first-seed models required')
    OUT.mkdir(parents=True);results={}
    for name in NAMES:
        manifest=json.loads((DATA/name/'manifest.json').read_text('utf8'))
        # Resolve input from the frozen original manifest before applying the guard.
        original=json.loads((Path(split['piece_roots'][name])/'manifest.json').read_text('utf8'))
        pcd=Path(original['record']['input']).resolve();input_hash=sha256(pcd)
        model_hashes={target:sha256(path) for target,path in model_paths.items()}
        reads=[];controls=[];forbidden=[Path(p).resolve() for p in ('DataSet','weld_pipeline/data','weld_pipeline/reports')]
        original_open=(builtins.open,io.open,Path.open)
        def guard(opener):
            def checked(path,mode='r',*args,**kwargs):
                if isinstance(path,(str,bytes,Path)) and ('r' in str(mode) or '+' in str(mode)):
                    file=Path(path).resolve()
                    if file==pcd:reads.append(str(file))
                    elif any(file.is_relative_to(root) for root in forbidden) or file.suffix.lower() in ('.stp','.step','.xlsx','.xls') or file.name.endswith('_welds.json'):
                        raise RuntimeError('Forbidden material runtime reference/cache: '+str(file))
                return opener(path,mode,*args,**kwargs)
            return checked
        started=time.perf_counter()
        builtins.open,io.open,Path.open=tuple(guard(opener) for opener in original_open)
        try:
            for opener in (builtins.open,io.open,Path.open):
                try:opener(Path('weld_pipeline/data/material_negative_control.step'),'r')
                except RuntimeError:controls.append(True)
                else:raise ValueError('Material reference-read negative control failed')
            models={target:joblib.load(path) for target,path in model_paths.items()}
            xyz=read_xyz_pcd(pcd)*1000.;sequences,geometry=make_mixed_sequences(xyz)
            extractor=SignedMaterialSupport(xyz)
            raw=[];keys=[];normal_cases=[]
            for si,sequence in enumerate(sequences):
                if sequence.get('curve_kind')=='arc':continue
                indices=np.unique(np.round(np.array([0.,.25,.5,.75,1.])*(len(sequence['xyz'])-1)).astype(int))
                block=extractor.describe_sequence(sequence,indices);raw.extend(block)
                keys.extend([(si,int(i)) for i in indices])
                if len(normal_cases)<12:
                    i=int(indices[len(indices)//2]);q=sequence['xyz'][i]
                    normals=sequence['features'][i,60:66].reshape(2,3)
                    expected=extractor.describe(q,normals,sequence['direction'])
                    views=frame_views(expected)
                    for exchange in (False,True):
                        ns=normals[::-1] if exchange else normals
                        for a,b in ((1.,1.),(-1.,1.),(1.,-1.),(-1.,-1.)):
                            f=extractor.describe(q,ns*np.array([a,b])[:,None],sequence['direction'])
                            error=min(float(np.abs(f-v).max()) for v in views)
                            normal_cases.append(error)
                            if error>1e-6:raise ValueError('Actual material frame permutation differs')
            raw=np.asarray(raw,np.float32)
            probabilities={target:predict_equivalent_views(model,raw) for target,model in models.items()}
        finally:
            builtins.open,io.open,Path.open=original_open
        # Labels and feature caches are read only after guarded inference has ended.
        expected=np.load(DATA/name/'signed.npy',allow_pickle=False)
        expected_keys=[(r['sequence'],r['sample_index']) for r in manifest['entries']]
        keys_equal=keys==expected_keys
        features_equal=keys_equal and np.array_equal(raw,expected)
        if not features_equal:raise ValueError('Fresh candidate material features differ from training: '+name)
        probability_error=max(float(np.abs(probabilities[t]-predict_equivalent_views(models[t],expected)).max()) for t in models)
        for changed in frame_views(raw[::max(1,len(raw)//12)]):
            for target,model in models.items():
                p=predict_equivalent_views(model,changed)
                reference=predict_equivalent_views(model,raw[::max(1,len(raw)//12)])
                if np.abs(p-reference).max()>1e-12:raise ValueError('Averaged model not equivalent-frame invariant')
            if not np.array_equal(side_sorted(changed),side_sorted(raw[::max(1,len(raw)//12)])):
                raise ValueError('Physical ablation frame invariant failed')
        unchanged=sha256(pcd)==input_hash and all(sha256(model_paths[t])==model_hashes[t] for t in model_paths)
        passed=bool(reads and len(controls)==3 and features_equal and probability_error<=1e-12 and unchanged)
        result=dict(piece=name,rows=len(raw),fresh_PCD_reads=len(reads),negative_controls=len(controls),
            training_feature_values_exact=features_equal,candidate_query_order_exact=keys_equal,
            prediction_replay_max_error=probability_error,normal_frame_cases=len(normal_cases),
            normal_frame_max_error=max(normal_cases),input_and_models_unchanged=unchanged,
            pcd_sha256=input_hash,model_sha256=model_hashes,elapsed_seconds=time.perf_counter()-started,
            all_checks_passed=passed,runtime_STEP_used=False,runtime_labels_used=False,runtime_preprocessing_cache_used=False,
            material_identity_reviewed=False,geometry_prediction_not_changed=True,full_contract_pass=False)
        np.savez(OUT/(name+'_diagnostic_probabilities.npz'),**probabilities)
        (OUT/(name+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
        results[name]=result
        print('MATERIAL_RAW_VERIFICATION',name,'ROWS',len(raw),'PASSED',passed,flush=True)
        if not passed:raise ValueError('Material raw PCD verification failed')
    complete=dict(pieces=results,all_checks_passed=True,verifier_sha256=sha256(__file__),
        source_sha256={p:sha256(p) for p in ('weld_pipeline/signed_material_support.py','weld_pipeline/mixed_sequences.py')},
        observed_check_read=False,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')


if __name__=='__main__':
    main()
