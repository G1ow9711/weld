"""Zero-fit rigid-coordinate intervention on frozen feature tables; not extraction scoring."""
import argparse
import gc
import os
import pickle
import shutil
import time
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, utc, check_resources


def worker(dest):
    import numpy as np
    from weld_pipeline.sjg_joint_object_features_1106 import side_rows
    started = time.perf_counter()
    model_file = OUT/'baseline_run302_20261009/run302/model.pkl'
    expected = read(OUT/'baseline_run302_20261009/manifest.json')['model_sha256']
    assert digest(model_file) == expected
    check_resources()
    with model_file.open('rb') as f:
        models = pickle.load(f)
    classifier, endpoint = models['object'], models['endpoint']
    positive = np.flatnonzero(classifier.classes_ != 0)
    rotations = {'identity': np.eye(3), 'translate_x_100mm': np.eye(3),
                 'translate_y_100mm': np.eye(3), 'translate_z_100mm': np.eye(3),
                 'rotate_z_90deg': np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])}
    results, distributions = [], []
    for key, prior_run in [('21','run303'),('BM79A_native_local','run304'),('10','run305'),('64','run306'),
                           ('4',None),('25',None),('36',None),('42',None)]:
        check_resources()
        feature_run = 'run293' if key == '64' else 'run308' if key in ('4','25','36','42') else 'run287'
        alias_run = 'run293' if key == '64' else 'run288'
        feature_file = OUT/feature_run/f'source_{key}/object_features.npy'
        fr = read(feature_file.parent/'receipt.json')
        assert digest(feature_file) == fr['features_sha256']
        alias_file = OUT/alias_run/f'source_{key}_alias_map.npz'
        ar = next(v for v in read(OUT/alias_run/'all_alias_maps_frozen_before_labels.json')['cases'] if v['source'] == key)
        assert digest(alias_file) == ar['map_sha256']
        with np.load(alias_file) as a:
            canonical = a['canonical_fragments']
        base = np.load(feature_file,mmap_mode='r')
        def quantile(v): return np.quantile(v,[0,.25,.5,.75,1]).tolist()
        d = dict(source=key,canonical_fragments=len(canonical),feature_sha256=fr['features_sha256'],
                 line_fraction=float(base[canonical,112].mean()),
                 fragment_length_mm_quantiles=quantile(np.expm1(base[canonical,80])),
                 face_angle_degrees_quantiles=quantile(np.degrees(np.arccos(np.clip(base[canonical,105],-1,1)))),
                 xyz_candidate_midpoint_mm_min=base[canonical,90:93].min(0).astype(float).tolist(),
                 xyz_candidate_midpoint_mm_max=base[canonical,90:93].max(0).astype(float).tolist(),
                 both_face_support_known_fraction=float(base[canonical,119].mean()))
        d['xyz_candidate_midpoint_mm_min'] = (np.array(d['xyz_candidate_midpoint_mm_min'])*1000).tolist()
        d['xyz_candidate_midpoint_mm_max'] = (np.array(d['xyz_candidate_midpoint_mm_max'])*1000).tolist()
        distributions.append(d)
        if prior_run is None:
            del base,canonical
            gc.collect()
            continue
        original_nodes = read(OUT/prior_run/'selected_graph.json')['nodes']
        original_rows = np.array([n['object_row'] for n in original_nodes],np.int64)
        rows = np.arange(2*len(canonical)) if key == '64' else original_rows
        # TRAIN uses all frozen admitted nodes; DEV uses every canonical object row.
        x = side_rows(base,canonical[rows//2],rows%2)
        probabilities, ratios = {}, {}
        for name, rotation in rotations.items():
            check_resources()
            y = x.copy()
            if name.startswith('translate'):
                axis = {'x':0,'y':1,'z':2}[name.split('_')[1]]
                for first in (84,87,90): y[:,first+axis] += np.float32(.1)
            elif name != 'identity':
                for first in (84,87,90,93,96,99,102): y[:,first:first+3] = y[:,first:first+3]@rotation.T
            p = classifier.predict_proba(y)[:,positive].sum(1)
            q = endpoint.predict(y)
            probabilities[name], ratios[name] = p,q
            del y
        p0, q0 = probabilities['identity'], ratios['identity']
        position = {int(row):i for i,row in enumerate(rows)}
        for node in original_nodes:
            assert abs(float(p0[position[node['object_row']]])-node['probability']) < 1e-12
        assert int((p0>=.5).sum()) == len(original_nodes)
        length = np.expm1(x[:,80].astype(np.float64))
        experiments = []
        for name in rotations:
            p,q = probabilities[name],ratios[name]
            # Changes in ratios map to travel distance on the same unchanged carrier.
            shift = np.max(np.abs(q-q0),axis=1)*length
            mask = p0>=.5
            experiments.append(dict(transform=name,admitted_objects=int((p>=.5).sum()),
                 original_admitted_retained=int(((p>=.5)&mask).sum()),
                 original_admitted_lost=int(((p<.5)&mask).sum()),
                 new_admitted=int(((p>=.5)&~mask).sum()),
                 original_admitted_endpoint_parameter_shift_mm_max=float(shift[mask].max(initial=0)),
                 original_admitted_endpoint_shift_over3mm=int((shift[mask]>3).sum()),
                 maximum_probability_change=float(np.max(np.abs(p-p0)))))
        results.append(dict(source=key,scope='all_canonical_objects' if key=='64' else 'frozen_admitted_TRAIN_objects',
                            object_rows=len(rows),original_admitted=len(original_nodes),
                            identity_probability_reproduction=True,experiments=experiments))
        write(dest/'partial.json',dict(results=results,distributions=distributions))
        print('RIGID_DIAGNOSTIC_SOURCE',key,'ROWS',len(rows),'SECONDS',round(time.perf_counter()-started,3),flush=True)
        del base,canonical,x,probabilities,ratios
        gc.collect()
    groups = {'two_face_point_support':list(range(80)), 'position_xyz':list(range(84,93)),
              'tangents_and_normals':list(range(93,105)),
              'other_geometry_masks':[i for i in range(80,128) if not 84<=i<105]}
    importance = {m:{g:float(models[m].feature_importances_[ix].sum()) for g,ix in groups.items()}
                  for m in ('object','endpoint')}
    assert digest(model_file)==expected
    write(dest/'worker_receipt.json',dict(status='BASELINE_RIGID_FEATURE_DIAGNOSTIC_COMPLETE',
        results=results,distributions=distributions,impurity_importance_noncausal=importance,
        source_model_sha256=expected,source_model_unchanged=True,feature_transform_only=True,
        raw_preprocessing_rigid_equivariance_tested=False,fit_calls=0,GT_reads=0,STEP_reads=0,
        extraction_correct_rate=None,extraction_precision_mm=None,seconds=time.perf_counter()-started,utc=utc()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run324');ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();dest=OUT/args.run
    if args.worker:
        try:worker(dest)
        except BaseException:
            write(dest/'worker_failure.json',dict(error=traceback.format_exc(),utc=utc()));raise
        return
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    dest.mkdir(exist_ok=False)
    snapshot=dest/'source_snapshot';snapshot.mkdir()
    for file in (__file__,ROOT/'weld_pipeline/sjg_joint_object_features_1106.py',ROOT/'weld_pipeline/sjg_owned_watchdog_1101.py'):
        shutil.copy2(file,snapshot)
    write(dest/'registered_protocol.json',dict(status='REGISTERED_ZERO_FIT_RIGID_DIAGNOSTIC',
        source_model='run302',transformations=['identity','translate_x_100mm','translate_y_100mm','translate_z_100mm','rotate_z_90deg'],
        TRAIN_scope='all frozen admitted objects',DEV_scope='all canonical objects',
        threshold=.5,no_GT_or_STEP_reads=True,no_threshold_selection=True,wall_seconds=180,utc=utc()))
    command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
             'weld_pipeline.sjg_baseline_rigid_diagnostic_1133','--run',args.run,'--worker']
    result=supervise(command,dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='DIAGNOSTIC_INCOMPLETE',watchdog=result));raise RuntimeError('Diagnostic incomplete')
    receipt=read(dest/'worker_receipt.json');receipt['external_wall_seconds']=result['wall_seconds']
    write(dest/'receipt.json',receipt)
    print('ZERO_FIT_DIAGNOSTIC_COMPLETE',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
