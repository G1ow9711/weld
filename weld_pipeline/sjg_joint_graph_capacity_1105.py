"""One complete TRAIN capacity gate for the finite fragment graph.

All seven pure-input geometric graphs are frozen BEFORE target reads. Joint
binary feasibility assigns every child one compatible original-curve witness;
two explicit unresolved side hypotheses are available per input fragment.
The independent D3 scorer then checks every decoded complete parent. This is
teacher-assisted capacity, never a prediction or extraction-accuracy result.
"""
import argparse
import gc
import json
import sys
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc, check_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph, specification

TRAIN_KEYS = ('21', '4', '10', '25', '36', '42', 'BM79A_native_local')


class ContinuousGraphBank:
    def __init__(self, graph): self.graph = graph
    def get(self, pid):
        return dict(curve=self.graph.curve(pid), relation=None, side_hypothesis=None)
    def coordinate(self, pid, interval, fraction):
        if int(interval) != 0 or not 0 <= float(fraction) <= 1: raise ValueError('Invalid continuous endpoint')
        return float(fraction) * self.graph.curve(pid).length


def choice(a, child, graph, connection='UNKNOWN'):
    length = graph.curve(a['profile']).length
    return dict(a, intervals=[0, 0], fractions=[a['lo']/length, a['hi']/length],
                weld_type=child['weld_type'], side='UNKNOWN', connection=connection)


def decode_parent(alternatives, parent, graph, query=0):
    from weld_pipeline.sjg_parent_set_runtime_1081 import decode_choices
    children = [choice(a, c, graph, 'UNKNOWN' if i == 0 else parent['gaps'][i-1])
                for i, (a, c) in enumerate(zip(alternatives, parent['children']))]
    return decode_choices([dict(query=query, probability=1., closed=parent['closed'],
                               child_count=len(children), children=children)], ContinuousGraphBank(graph))


def freeze_graphs(dest):
    rows = []
    write(dest/'feature_and_graph_specification.json', specification())
    for key in TRAIN_KEYS:
        check_resources(); t = time.perf_counter(); base = OUT/f'run185/source_{key}'
        binding = read(base/'input_freeze.json')
        for name in ('profiles.json', 'surfaces.json', 'surface_support.npz'):
            assert digest(base/name) == binding['files'][name]
        profiles = read(base/'profiles.json'); graph = JointSegmentGraph.build(profiles)
        path = dest/f'source_{key}_geometry.npz'
        np.savez(path, offsets=graph.offsets, spans=graph.spans)
        row = dict(source=key, profiles=len(profiles), fragments=len(graph.spans),
                   side_states=2, directed_states_per_side=2, source_files=binding['files'],
                   geometry_file=path.name, geometry_sha256=digest(path),
                   target_reads=0, seconds=time.perf_counter()-t)
        rows.append(row); print('FINITE_GRAPH_FROZEN', key, 'FRAGMENTS', len(graph.spans), flush=True)
        del profiles, graph; gc.collect()
    write(dest/'all_geometry_frozen_before_labels.json', dict(cases=rows, target_reads=0,
          code_sha256=digest(__import__('pathlib').Path(__file__).with_name('sjg_joint_segment_graph_1104.py')),
          specification_sha256=digest(dest/'feature_and_graph_specification.json'), utc=utc()))
    return rows


def solve_case(key, graph, compiled, native_target, directory):
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    from weld_pipeline.contract_geometry_d3_409 import child_error
    from weld_pipeline.world_set_evaluation_285 import reference_segment
    from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets, score_native_parents
    from weld_pipeline.sjg_parent_set_runtime_1081 import child_geometry
    assert 'torch' not in sys.modules, 'CPU capacity must not import Torch indirectly'
    native = native_targets(native_target)
    variables = []; groups = []; occupancy = {}; unsupported = []; invalid = []
    objective = []; constraints = []; lower = []; upper = []
    parent_groups = []; curve_ends = {}
    refs = [[reference_segment(c) for c in p['children']] for p in native['parents']]

    def add_constraint(indices, lo, hi):
        constraints.append(indices); lower.append(lo); upper.append(hi)

    for pi, parent in enumerate(compiled['parents']):
        pg = []
        for ci, child in enumerate(parent['children']):
            ids = []
            for ai, alt in enumerate(child['alternatives']):
                try:
                    fragments = graph.witness_fragments(alt)
                    if any(not graph.connection(a,b,'SAME_CHILD',alt['forward'],alt['forward'])
                           for a,b in zip(fragments[:-1], fragments[1:])):
                        raise ValueError('Missing internal SAME_CHILD connection')
                    standalone = dict(children=[child], gaps=[], closed=bool(alt['closed']))
                    dec = decode_parent([alt], standalone, graph)
                    if dec['invalid_predictions']: raise ValueError(dec['invalid_predictions'][0]['reason'])
                    error = child_error(dec['parents'][0]['segments'][0], native['parents'][pi]['children'][ci],
                                        refs[pi][ci], bool(alt['closed']), False)
                    if error is None: raise ValueError('Original finite witness fails standalone 3 mm geometry')
                except ValueError as exc:
                    invalid.append(dict(parent=pi, child=ci, alternative=ai, profile=alt['profile'], reason=str(exc)))
                    continue
                vid = len(variables); ids.append(vid)
                variables.append(dict(parent=pi,child=ci,alternative=ai,alt=alt,fragments=fragments))
                for fragment in fragments: occupancy.setdefault(fragment, []).append(vid)
                rows = graph.spans[fragments]; widths = rows[:,1]-rows[:,0]
                completeness = float(np.dot(widths, rows[:,2])/widths.sum())
                # Only input support and stable input order rank witnesses. GT
                # geometry is a feasibility gate, not an identification feature.
                objective.append(1.-completeness+int(alt['profile'])*1e-12)
                points, _ = child_geometry(graph.curve(alt['profile']),alt['lo'],alt['hi'],
                                            alt['forward'],alt['closed'],alt['wrap'])
                curve_ends[vid] = points[[0,-1]]
            if not ids:
                unsupported.append(dict(parent=pi,child=ci,original_child_id=child['original_child_id'],
                    reason='No finite fragment path remains', alternatives=len(child['alternatives'])))
            groups.append(ids); pg.append(ids); add_constraint(ids, 1., 1.)
        parent_groups.append(pg)
    if unsupported:
        return dict(status='CAPACITY_MISSING_CHILD_PATH', first_missing=unsupported[0],
                    unsupported=unsupported, rejected_witnesses=invalid)
    # At most one occupancy per side. Since each TRAIN witness is a contiguous
    # original-curve interval, capacity <=2 admits exact two-color assignment.
    for ids in occupancy.values():
        if len(ids)>2: add_constraint(ids, -np.inf, 2.)
    incompatible = []; required_pairs = 0
    for pi, (parent, pg) in enumerate(zip(compiled['parents'], parent_groups)):
        for ci in range(1,len(pg)):
            kind = parent['gaps'][ci-1]; count = 0
            for a in pg[ci-1]:
                va = variables[a]
                for b in pg[ci]:
                    vb = variables[b]; aa, bb = va['alt'], vb['alt']
                    legal = graph.connection(va['fragments'][-1], vb['fragments'][0], kind,
                                             aa['forward'],bb['forward'])
                    if legal:
                        pair = dict(children=parent['children'][ci-1:ci+1],gaps=[kind],closed=False)
                        decoded = decode_parent([aa,bb],pair,graph)
                        legal = not decoded['invalid_predictions']
                        if legal:
                            segment = decoded['parents'][0]['segments'][1]
                            legal = child_error(segment,native['parents'][pi]['children'][ci],
                                                refs[pi][ci],False,False) is not None
                    if legal: count += 1
                    else: add_constraint([a,b],-np.inf,1.); incompatible.append([a,b])
            required_pairs += 1
            if not count:
                return dict(status='CAPACITY_MISSING_PARENT_CONNECTION',first_missing=dict(parent=pi,
                    previous_child=ci-1,next_child=ci,connection=kind,
                    previous_profiles=[variables[x]['alt']['profile'] for x in pg[ci-1]],
                    next_profiles=[variables[x]['alt']['profile'] for x in pg[ci]]),
                    rejected_witnesses=invalid)
    ri=[]; col=[]
    for i, ids in enumerate(constraints): ri.extend([i]*len(ids)); col.extend(ids)
    matrix = coo_matrix((np.ones(len(col)),(ri,col)),shape=(len(constraints),len(variables))).tocsc()
    # HiGHS expects int32 sparse indices in this pinned SciPy build.
    matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32)
    result=milp(np.asarray(objective),integrality=np.ones(len(variables),np.int32),
        bounds=Bounds(0.,1.),constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper)),
        options=dict(time_limit=30.,presolve=True))
    if result.status != 0 or result.x is None:
        return dict(status='CAPACITY_JOINT_ASSIGNMENT_FAILED', solver_status=int(result.status),
                    solver_message=str(result.message), all_children=len(groups), variables=len(variables),
                    occupancy_constraints=sum(len(v)>2 for v in occupancy.values()),
                    first_missing=dict(reason='No certified joint assignment; do not treat independent child witnesses as joint capacity'))
    selected = [int(i) for i in np.flatnonzero(result.x>.5)]
    assert len(selected)==len(groups)
    by_child={(variables[i]['parent'],variables[i]['child']):i for i in selected}
    # Explicitly realize side slots and verify no fragment is used twice on one
    # side. No side truth is inferred from this oracle coloring.
    side_for={}; occupied={}
    selected.sort(key=lambda i:(variables[i]['alt']['profile'],min(variables[i]['fragments']),
                                max(variables[i]['fragments']),i))
    for i in selected:
        fragments=variables[i]['fragments']
        available=[s for s in range(2) if all((f,s) not in occupied for f in fragments)]
        if not available:
            return dict(status='CAPACITY_SIDE_ASSIGNMENT_FAILED', first_missing=variables[i],
                        reason='Joint count relaxation could not be realized with two side hypotheses')
        side_for[i]=available[0]
        for f in fragments: occupied[f,available[0]]=i
    decoded=dict(parents=[],invalid_predictions=[],total_prediction_children=0); witnesses=[]
    for pi,parent in enumerate(compiled['parents']):
        ids=[by_child[pi,ci] for ci in range(parent['child_count'])]
        alternatives=[variables[i]['alt'] for i in ids]
        dec=decode_parent(alternatives,parent,graph,pi)
        decoded['parents'].extend(dec['parents']);decoded['invalid_predictions'].extend(dec['invalid_predictions'])
        decoded['total_prediction_children']+=dec['total_prediction_children']
        witnesses.append(dict(parent=pi,children=[dict(variables[i],side_hypothesis=side_for[i]) for i in ids],
                              gaps=parent['gaps'],closed=parent['closed']))
    score=score_native_parents(decoded,native_target,[False]*len(compiled['parents']))
    overall=score['categories']['overall']; passed=overall['TP']==compiled['G'] and overall['FP']==0
    write(directory/f'source_{key}_joint_witnesses.json',dict(witnesses=witnesses,score=score,
          decoded=decoded,side_truth_verified=False,all_runtime_edges_materialized=False))
    return dict(status='COMPLETE_JOINT_GEOMETRIC_CAPACITY_PASSED' if passed else 'COMPLETE_PARENT_GEOMETRY_FAILED',
        G=compiled['G'],parents=len(compiled['parents']),variables=len(variables),fragments=len(graph.spans),
        joint_occupancies=len(occupied),required_child_connections=required_pairs,
        rejected_witnesses=invalid,oracle_geometry_score=overall,
        first_missing=None if passed else dict(reason='Selected complete assignment failed independent D3',matches=score['matches']),
        solver_status=int(result.status), side_truth_verified=False, optimizer_updates=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run285');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);start=time.perf_counter();rows=[]
    try:
        freezes=freeze_graphs(dest)
        targets=read(OUT/'run264/receipt.json'); cohort=read(OUT/'run98/local_cohort.json')
        for frozen in freezes:
            key=frozen['source'];check_resources(); t=time.perf_counter()
            original=next(r for r in targets['cases'] if r['source']==key)
            compiled_path=OUT/f'run264/source_{key}_targets.json'
            assert digest(compiled_path)==original['compiled_sha256']
            case=next(r for r in cohort['cases'] if r['source_key']==key)
            assert digest(OUT/case['target_file'])==case['target_sha256']
            assert digest(dest/frozen['geometry_file'])==frozen['geometry_sha256']
            profiles=read(OUT/f'run185/source_{key}/profiles.json')
            with np.load(dest/frozen['geometry_file']) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
            row=solve_case(key,graph,read(compiled_path),read(OUT/case['target_file']),dest)
            row.update(source=key,seconds=time.perf_counter()-t);rows.append(row)
            write(dest/'capacity_progress.json',dict(cases=rows,utc=utc()))
            print('JOINT_CAPACITY',key,row['status'],row.get('oracle_geometry_score'),flush=True)
            if row['status']!='COMPLETE_JOINT_GEOMETRIC_CAPACITY_PASSED':break
            del graph,profiles;gc.collect()
        passed=len(rows)==7 and all(r['status']=='COMPLETE_JOINT_GEOMETRIC_CAPACITY_PASSED' for r in rows)
        write(dest/'receipt.json',dict(status='JOINT_GEOMETRIC_CAPACITY_PASSED' if passed else 'JOINT_GEOMETRIC_CAPACITY_FAILED',
            cases=rows,required_parents=222,required_children=363,
            tested_children=sum(r.get('G',0) for r in rows),
            untested_sources=[k for k in TRAIN_KEYS if k not in [r['source'] for r in rows]],
            extraction_correct_rate=None,extraction_precision_mm=None,scientific_training=False,
            oracle_not_model_accuracy=True,all_features_materialized=False,all_runtime_edges_materialized=False,
            Torch_imported='torch' in sys.modules,DEV_DIAG_labels_read=0,seconds=time.perf_counter()-start,utc=utc()))
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,
                                      scientific_training=False,utc=utc()));raise


if __name__=='__main__':main()
