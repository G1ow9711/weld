"""Frozen complete parents plus learned interval validity and duplicate checks.

Point inputs are prepared before any TRAIN target access. Original best-model
parents are immutable. Full-parent compatibility trains the additions; this
does not certify weld attributes, material roles, or independent acceptance.
"""
import argparse
import gc
import json
import pickle
import time
import traceback
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_shared_prediction_reader_1236 import read_shared_predictions
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations

TRAIN = ('21', 'BM79A_native_local', '10')
KEYS = (*TRAIN, '64')
SCHEMA = 'FROZEN379_PLUS408_PARENT_VALIDITY_AND_SYMMETRIC_DUPLICATES_V1'


def source(key, route=None):
    route = route or {}
    old = OUT / route.get('baseline_run', 'run379') / f'test_{key}'
    new = OUT / route.get('proposal_run', 'run408') / f'proposal_{key}'
    br = read(old / 'final_prediction.json')
    nr = read(new / 'prediction_receipt.json')
    assert digest(old / 'final_decoded.json') == br['decoded_sha256']
    assert digest(new / 'decoded.json') == nr['decoded_sha256']
    assert br['GT_reads'] == nr['GT_reads'] == br['STEP_reads'] == nr['STEP_reads'] == 0
    baseline = read_shared_predictions(old / 'final_decoded.json')
    additions = read_shared_predictions(new / 'decoded.json')
    metadata = read(old / 'parent_metadata.json')
    selected = br['selected_candidate_indices']
    bm = [metadata[i] for i in selected]
    assert len(bm) == len(baseline['parents'])
    return baseline, additions, bm, dict(baseline_sha256=br['decoded_sha256'], new_sha256=nr['decoded_sha256'])


def prepare(dest, keys=KEYS, routes=None):
    from weld_pipeline.sjg_joint_pair_model_1248 import load_inputs
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    routes = routes or {}
    for key in keys:
        check_resources(); begin = time.perf_counter()
        route = routes.get(key, {})
        baseline, additions, bm, binding = source(key, route)
        intervals = read(OUT / route.get('proposal_run', 'run408') / f'proposal_{key}/selected_intervals.json')['intervals']
        assert not additions['invalid_predictions']
        assert len(intervals) == len(additions['parents'])
        s = load_inputs(key, **route.get('pair_inputs', {'input_run': 'run402'})); case = load_case(key); graph = case['graph']
        case['x']._mmap.close()
        row_lookup = {int(row): i for i, row in enumerate(s['m']['rows'])}
        xx, metadata = [], []
        for interval, parent in zip(intervals, additions['parents']):
            row = interval['row']; i = row_lookup[row]
            a, b = map(int, s['m']['offsets'][i:i+2]); n = b-a
            q = s['m']['q_mm'][a:b]
            left = int(np.argmin(abs(q-interval['lo_mm'])))
            right = int(np.argmin(abs(q-interval['hi_mm'])))
            assert q[left] == interval['lo_mm'] and q[right] == interval['hi_mm'] and left < right
            curve = graph.curve(row//2); length = q[right]-q[left]
            mean_occupancy = (s['prefix'][a+right]-s['prefix'][a+left])/length
            category = interval['category']; assert 1 <= category <= 4
            extra = [np.log1p(length/10), length/curve.length, mean_occupancy,
                     interval['margin'], *[float(category == c) for c in (1,2,3,4)]]
            xx.append(np.r_[s['x'][2*a+left], s['x'][2*a+n+right], s['context'][row], extra])
            faces = graph.profiles[row//2].get('face_indices', [])[:2]
            faces = faces[::-1] if row % 2 else faces
            metadata.append(dict(graph_nodes=[], face_pairs=[faces], physical_sides=['UNKNOWN'],
                side_known=False, probability_mean=float(1/(1+np.exp(-min(interval['margin'],80)))),
                length_mm=sum(c['length_mm'] for c in parent['segments']),
                line_fraction=float(parent['segments'][0]['kind'] == 'line')))
        x = np.asarray(xx, np.float32).reshape(-1, 528)
        assert np.isfinite(x).all()
        folder = dest / f'source_{key}'; folder.mkdir()
        np.save(folder / 'features.npy', x)
        write(folder / 'metadata.json', bm+metadata)
        write(folder / 'receipt.json', dict(source=key, schema=SCHEMA, **binding,
            feature_sha256=digest(folder / 'features.npy'), metadata_sha256=digest(folder / 'metadata.json'),
            base_count=len(bm), additions=len(metadata), columns=528, GT_reads=0, STEP_reads=0,
            endpoint_feature_sha256=s['receipt']['feature_sha256'], context_sha256=s['context_sha256'],
            seconds=time.perf_counter()-begin))
        s['x']._mmap.close(); s['context']._mmap.close()
        del s, case, graph, x, xx, baseline, additions, bm, metadata
        gc.collect()
        print('PURE_INPUT_READY', key, flush=True)
    write(dest / 'inputs_frozen_before_train_labels.json', dict(sources=list(keys), GT_reads=0,
        routes=routes, receipts={k:digest(dest/f'source_{k}/receipt.json') for k in keys}))


def load_context(dest, key):
    folder = dest / f'source_{key}'; r = read(folder / 'receipt.json')
    assert digest(folder / 'features.npy') == r['feature_sha256']
    assert digest(folder / 'metadata.json') == r['metadata_sha256']
    baseline, additions, _, binding = source(key)
    assert all(r[k] == v for k, v in binding.items())
    parents = baseline['parents'] + additions['parents']
    metadata = read(folder / 'metadata.json')
    assert len(parents) == len(metadata)
    return baseline, additions, ParentRelations(parents, metadata), r


def endpoint_distances(parents):
    ends = np.asarray([[p['segments'][0]['points'][0], p['segments'][-1]['points'][-1]] for p in parents], float)
    distances = np.empty((len(parents), len(parents)), np.float32)
    for i, ep in enumerate(ends):
        distances[i] = np.minimum(np.linalg.norm(ends-ep, axis=2).max(1),
                                  np.linalg.norm(ends-ep[::-1], axis=2).max(1))
    return distances


def pair_features(context, a, b):
    first = context.pair(a, b); second = context.pair(b, a)
    return np.r_[.5*(first+second), abs(first-second)].astype(np.float32)


def candidate_pair_possible(context, distance, a, b):
    pa, pb = context.parents[a], context.parents[b]
    if pa['child_count'] != pb['child_count'] or pa['is_closed'] != pb['is_closed']:
        return False
    # Closed parents may use arbitrary anchors/cyclic child ordering.
    # Open duplicate candidates each within 3 mm of one truth have ends <=6 mm apart.
    return bool(pa['is_closed'] or distance[a, b] <= 6.00001)


def label(dest):
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for
    from weld_pipeline.sjg_parent_compatibility_1161 import compile_compatibility
    assert set(read(dest / 'inputs_frozen_before_train_labels.json')['sources']) == set(KEYS)
    cases = []
    for key in TRAIN:
        begin = time.perf_counter(); baseline, additions, context, r = load_context(dest, key)
        target, binding = target_for(key)
        combined = dict(parents=context.parents, invalid_predictions=[],
                        total_prediction_children=sum(p['child_count'] for p in context.parents))
        compatible = compile_compatibility(combined, target)
        edges = [set(v) for v in compatible['compatible_targets']]
        offset = r['base_count']; known = {v['row']: v['complete_background_supervision_allowed']
            for v in read(OUT / f'run406/source_{key}/compatibility.json')['rows']}
        intervals = read(OUT/f'run408/proposal_{key}/selected_intervals.json')['intervals']
        y = np.array([1 if edges[offset+i] else (0 if known.get(v['row'], False) else -1)
                      for i, v in enumerate(intervals)], np.int8)
        distances = endpoint_distances(context.parents)
        eligible = [i for i, v in enumerate(edges) if v]
        selected_pairs = set()
        for a in eligible:
            negatives = []
            for b in eligible:
                if b == a: continue
                duplicate = len(edges[a] | edges[b]) == 1
                if candidate_pair_possible(context, distances, a, b):
                    selected_pairs.add(tuple(sorted((a,b))))
                if not duplicate:
                    negatives.append((float(distances[a,b]), b))
            # TRAIN-only nearest nonduplicate negatives. No inference top-k limit.
            for _, b in sorted(negatives)[:4]: selected_pairs.add(tuple(sorted((a,b))))
        px, py, pairs = [], [], []
        for a, b in sorted(selected_pairs):
            features = pair_features(context, a, b)
            assert np.array_equal(features, pair_features(context, b, a))
            px.append(features); py.append(int(len(edges[a] | edges[b]) == 1)); pairs.append([a,b])
        folder = dest / f'source_{key}'
        np.savez(folder / 'training_targets.npz', validity=y,
                 duplicate_x=np.asarray(px,np.float32).reshape(-1,32), duplicate_y=np.asarray(py,np.int8))
        write(folder / 'compatibility.json', dict(compatible, source=key, target_sha256=binding['target_sha256']))
        write(folder / 'duplicate_training_pairs.json', dict(pairs=pairs,
            positives=sum(py), negatives=len(py)-sum(py), symmetric_controls_passed=True,
            ambiguous_geometric_aliases_allow_two_distinct_GT_assignments=True))
        receipt = dict(source=key, G=target['original_children'], known_positive=int((y==1).sum()),
            known_negative=int((y==0).sum()), unknown=int((y<0).sum()), duplicate_positive=sum(py),
            duplicate_negative=len(py)-sum(py), maximum_coverable_children=compatible['maximum_matched_children'],
            targets_sha256=digest(folder / 'training_targets.npz'),
            compatibility_sha256=digest(folder / 'compatibility.json'), seconds=time.perf_counter()-begin)
        cases.append(receipt); print('TRAIN_PARENT_LABELS', receipt, flush=True)
        del context, distances; gc.collect()
    assert sum(c['G'] for c in cases) == 213
    write(dest / 'supervision_receipt.json', dict(cases=cases, G=213, DEV_reads=0,
        original_targets_unchanged=True, full_contract_pass=False))


def fit(dest, input_dest):
    if (input_dest/'validity_scope_conflict.json').exists():
        raise RuntimeError('Known child/parent validity scope conflict: separate child validity and parent assembly before fitting again')
    begin = time.perf_counter(); receipt = read(input_dest/'supervision_receipt.json')
    assert read(input_dest/'prefit_audit.json')['status']=='PASS'
    models = {}; traces = {}
    for name in ('validity', 'duplicate'):
        xs, ys, ws = [], [], []
        for case in receipt['cases']:
            key = case['source']; folder = input_dest/f'source_{key}'
            assert digest(folder/'training_targets.npz') == case['targets_sha256']
            source_receipt = read(folder/'receipt.json')
            assert digest(folder/'features.npy') == source_receipt['feature_sha256']
            with np.load(folder/'training_targets.npz') as z:
                if name == 'validity':
                    x = np.load(folder/'features.npy'); y = z['validity']
                else: x = z['duplicate_x']; y = z['duplicate_y']
                mask = y >= 0; x = x[mask]; y = y[mask]
            if not len(y): continue
            w = np.zeros(len(y)); classes = np.unique(y)
            for c in classes: w[y==c] = 1/(len(TRAIN)*len(classes)*int((y==c).sum()))
            xs.append(x); ys.append(y); ws.append(w)
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        assert set(y.tolist()) == {0,1} and np.isfinite(x).all()
        model = ExtraTreesClassifier(n_estimators=16, max_depth=16, min_samples_leaf=2,
            max_features=.75, bootstrap=False, random_state=1031, n_jobs=2, warm_start=True)
        trace=[]
        for batch in range(1,5):
            check_resources(); model.set_params(n_estimators=batch*16); model.fit(x,y,sample_weight=w)
            trace.append(dict(batch=batch, trees=len(model.estimators_), seconds=time.perf_counter()-begin))
            print('FOREST', name, 'BATCH', batch, '/4 TREES', len(model.estimators_), '/64', flush=True)
        model.set_params(n_jobs=1, warm_start=False); models[name]=model; traces[name]=trace
        # Fixed final model, no tree/threshold selection on DEV.
        p=model.predict_proba(x); assert np.array_equal(p,model.predict_proba(x[::-1])[::-1])
    payload=dict(schema=SCHEMA, models=models, source_run=input_dest.name, threshold=.5,
                 base_run='run379', proposal_run='run408')
    atomic_binary_write(dest/'precision_models.pkl', lambda f:pickle.dump(payload,f,protocol=5))
    with (dest/'precision_models.pkl').open('rb') as f: reloaded=pickle.load(f)
    assert np.array_equal(models['duplicate'].predict_proba(x), reloaded['models']['duplicate'].predict_proba(x))
    for case in receipt['cases']:
        xx=np.load(input_dest/f"source_{case['source']}/features.npy")
        assert np.array_equal(models['validity'].predict_proba(xx), reloaded['models']['validity'].predict_proba(xx))
    write(dest/'training_receipt.json', dict(status='ADDITIVE_COMPLETE_PARENT_MODELS_TRAINED',
        TRAIN=list(TRAIN), G=213, DEV_reads=0, STEP_reads=0, traces=traces,
        supervision_receipt_sha256=digest(input_dest/'supervision_receipt.json'),
        model_sha256=digest(dest/'precision_models.pkl'), frozen_best379_unchanged=True,
        threshold=.5, probability_calibration_not_established=True, full_contract_pass=False,
        seconds=time.perf_counter()-begin))


def infer(dest, key):
    begin=time.perf_counter(); r=read(dest/'training_receipt.json')
    assert digest(dest/'precision_models.pkl')==r['model_sha256']
    with (dest/'precision_models.pkl').open('rb') as f: payload=pickle.load(f)
    assert payload['schema']==SCHEMA
    input_dest=OUT/payload['source_run']; baseline, additions, context, source_r=load_context(input_dest,key)
    x=np.load(input_dest/f'source_{key}/features.npy')
    probability=payload['models']['validity'].predict_proba(x)[:,1]
    distances=endpoint_distances(context.parents)
    offset=source_r['base_count']; selected=list(range(offset)); accepted=[]; decisions=[]
    for local in sorted(range(len(probability)),key=lambda i:(-probability[i],i)):
        index=offset+local; p=float(probability[local]); duplicate=None
        if p > .5:
            close=[j for j in selected if candidate_pair_possible(context,distances,index,j)]
            if close:
                px=np.stack([pair_features(context,index,j) for j in close])
                q=payload['models']['duplicate'].predict_proba(px)[:,1]
                maximum=int(q.argmax())
                if q[maximum]>.5: duplicate=dict(selected_index=close[maximum],probability=float(q[maximum]))
            if duplicate is None: selected.append(index); accepted.append(local)
        decisions.append(dict(candidate_index=local,validity_probability=p,
                              duplicate_of=duplicate,selected=local in accepted))
    result=dict(baseline); result['parents']=baseline['parents']+[additions['parents'][i] for i in accepted]
    result['invalid_predictions']=baseline['invalid_predictions']+additions['invalid_predictions']
    result['total_prediction_children']=sum(p['child_count'] for p in result['parents'])+sum(v['counted_false_children'] for v in result['invalid_predictions'])
    result.update(runtime_model=SCHEMA, GT_reads=0, STEP_reads=0)
    folder=dest/f'test_{key}'; folder.mkdir()
    with (folder/'final_decoded.json').open('w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,allow_nan=False,separators=(',',':'),
                  default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())
    write(folder/'decisions.json', dict(decisions=decisions, added_indices=accepted))
    write(folder/'prediction_receipt.json', dict(status='PURE_INPUT_FINAL_PARENT_PREDICTION_FROZEN', source=key,
        model_sha256=r['model_sha256'], decoded_sha256=digest(folder/'final_decoded.json'),
        baseline_sha256=source_r['baseline_sha256'], baseline_parents=offset, added_parents=len(accepted),
        GT_reads=0, STEP_reads=0, baseline_geometry_and_invalid_ledger_preserved=True,
        unimplemented_material_role_exclusion=True, full_contract_pass=False, seconds=time.perf_counter()-begin))
    print('ADDITIVE_OUTPUT',key,'BASE',offset,'ADDED',len(accepted),'SECONDS',round(time.perf_counter()-begin,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--stage',choices=['prepare','label','train','infer'],required=True)
    ap.add_argument('--source');ap.add_argument('--input-run',default='run409');a=ap.parse_args()
    dest=OUT/a.run
    if a.stage in ('prepare','train'):
        dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json',dict(schema=SCHEMA, source_sha256=digest(__file__),
            stage=a.stage, TRAIN=list(TRAIN), DEV=['64'], original_G=213, trees_per_model=64,
            threshold=.5, preserve_best379=True, whole_parent_compatibility=True,
            no_independent_child_reversal=True, external_consultation_paused=True, full_contract_pass=False))
    try:
        if a.stage=='prepare':prepare(dest)
        elif a.stage=='label':label(dest)
        elif a.stage=='train':fit(dest,OUT/a.input_run)
        else:infer(dest,a.source)
    except BaseException:
        write(dest/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
