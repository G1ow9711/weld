"""Pro306: frozen DEV coverage audit; never a training or extraction score."""
import argparse,itertools,pickle,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_forest_train_1109 import load_case,attributes
from weld_pipeline.sjg_joint_graph_capacity_1105 import solve_case
from weld_pipeline.sjg_local_curve_supervision_915 import supervise,compatible_interval
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_link_runtime_1107 import pair_features
from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import variants,child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment


class CanonicalCapacityGraph(JointSegmentGraph):
    def witness_fragments(self,alternative):
        return [int(self.alias[f]) for f in super().witness_fragments(alternative)]


def joint_fixed_sides(options,directory,stage):
    """Same binary full-parent occupancy formulation, explicit allowed side slots."""
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import coo_matrix
    missing=[i for i,o in enumerate(options) if not o]
    if len(missing)==len(options):
        return dict(status='PROVEN_INFEASIBLE',reason='At least one complete parent has no legal alternative',
                    zero_alternative_parents=missing,selected=[],selected_parent_indices=[])
    rows=[];cols=[];lo=[];hi=[];variables=[];occupied={}
    for pi,alternatives in enumerate(options):
        if not alternatives:continue
        ids=[]
        for alt in alternatives:
            j=len(variables);variables.append(dict(parent=pi,**alt));ids.append(j)
            for slot in alt['slots']:occupied.setdefault(slot,[]).append(j)
        k=len(lo);rows.extend([k]*len(ids));cols.extend(ids);lo.append(1.);hi.append(1.)
    for ids in occupied.values():
        if len(ids)<2:continue
        k=len(lo);rows.extend([k]*len(ids));cols.extend(ids);lo.append(-np.inf);hi.append(1.)
    matrix=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(lo),len(variables))).tocsc()
    matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32)
    result=milp(np.zeros(len(variables)),integrality=np.ones(len(variables)),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,lo,hi),options=dict(time_limit=15.,presolve=True))
    subset_status='PROVEN_FEASIBLE' if result.status==0 else 'PROVEN_INFEASIBLE' if result.status==2 else 'NOT_PROVEN'
    status='PROVEN_INFEASIBLE' if missing else subset_status
    selected=[variables[i] for i in np.flatnonzero(result.x>.5)] if result.status==0 else []
    report=dict(status=status,subset_status=subset_status,zero_alternative_parents=missing,
                solver_status=int(result.status),solver_message=str(result.message),
                variables=len(variables),selected=selected,selected_parent_indices=[v['parent'] for v in selected])
    write(directory/f'{stage}_joint_assignment.json',report);return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run307');args=ap.parse_args()
    dest=OUT/args.run;start=time.perf_counter();rows=[]
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-start,utc=utc(),**extra))
        print('DEV_COVERAGE_PHASE',name,round(time.perf_counter()-start,3),extra,flush=True)
    def save():write(dest/'parent_progress.json',dict(rows=rows,new_fit_calls=0,not_extraction_accuracy=True,utc=utc()))
    try:
        phase('FREEZE_ALL_INPUT_ONLY_OBJECT_PREDICTIONS')
        original=read(OUT/'run302/receipt.json');model_path=OUT/'run302/model.pkl'
        assert digest(model_path)==original['model_sha256']
        with model_path.open('rb') as f:models=pickle.load(f)
        case=load_case('64',supervision=False);graph=case['graph'];x=case['x'];n=len(x)
        p=np.empty(n);category=np.empty(n,np.int16);ratios=np.empty((n,2));positive_cols=np.flatnonzero(models['object'].classes_!=0)
        for first in range(0,n,32768):
            last=min(first+32768,n);prob=models['object'].predict_proba(x[first:last])
            p[first:last]=prob[:,positive_cols].sum(1)
            category[first:last]=models['object'].classes_[positive_cols[prob[:,positive_cols].argmax(1)]]
            ratios[first:last]=np.sort(models['endpoint'].predict(x[first:last]),axis=1);check_resources()
        np.savez(dest/'all_object_predictions_before_DEV_labels.npz',probability=p,category=category,ratios=ratios)
        runtime=read(OUT/'run306/selected_graph.json');nodes=runtime['nodes']
        assert sorted(np.flatnonzero(p>=.5).tolist())==sorted(node['object_row'] for node in nodes)
        for node in nodes:
            row=node['object_row'];lo,hi=graph.spans[node['fragment'],:2]
            assert np.allclose(lo+ratios[row]*(hi-lo),[node['lo'],node['hi']],rtol=0,atol=1e-10)
        immutable={str(model_path):digest(model_path),str(OUT/'run306/decoded.json'):digest(OUT/'run306/decoded.json'),
                   str(OUT/'run306/selected_graph.json'):digest(OUT/'run306/selected_graph.json')}
        write(dest/'all_predictions_frozen_before_DEV_labels.json',dict(model_sha256=original['model_sha256'],
            prediction_array_sha256=digest(dest/'all_object_predictions_before_DEV_labels.npz'),
            object_rows=n,positive_objects=int((p>=.5).sum()),feature_receipt_sha256=case['feature_receipt_sha256'],
            alias_map_sha256=case['alias_map_sha256'],geometry_sha256=case['geometry_sha256'],DEV_labels_read=0,utc=utc()))
        phase('READ_DEV_TARGETS_ONLY_AFTER_FREEZE')
        metadata=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='64')
        assert digest(OUT/metadata['target_file'])==metadata['target_sha256'];target=read(OUT/metadata['target_file'])
        for i,parent in enumerate(target['parents']):
            rows.append(dict(parent_index=i,original_parent_id=parent['original_parent_id'],
                C0='NOT_PROVEN',C1='NOT_PROVEN',C2='NOT_PROVEN',failure_stage='NOT_PROVEN',reason='Audit pending'))
        save()
        # A single-child open-line fast proof is exact for this registered case;
        # other structures are left unproved rather than silently simplified.
        shape_ok=all(len(t['children'])==1 and not t['is_closed'] and t['children'][0]['original_kind']=='line' for t in target['parents'])
        if not shape_ok:raise RuntimeError('DEV parent shape is outside exact single-line audit; rows remain NOT_PROVEN')
        phase('C0_ALL_ORIGINAL_CURVES_AND_JOINT_OCCUPANCY')
        sup=supervise(graph.profiles,target);write(dest/'DEV_only_full_carrier_compatibility.json',sup)
        by_child={c['original_child_id']:[] for par in target['parents'] for c in par['children']}
        for pid,events in sup['by_profile'].items():
            length=graph.curve(int(pid)).length
            for e in events:
                by_child[e['original_child_id']].append(dict(profile=int(pid),lo=max(0.,e['lo']),hi=min(length,e['hi']),
                    forward=e['target_direction_forward'],closed=e['closed_whole'],wrap=e['wrap']))
        compiled=dict(source='64',G=target['original_children'],parents=[])
        for parent in target['parents']:
            c=parent['children'][0];compiled['parents'].append(dict(closed=False,child_count=1,gaps=[],children=[dict(
                original_child_id=c['original_child_id'],weld_type={'horizontal':0,'vertical':1}[c['weld_type']],
                alternatives=by_child[c['original_child_id']])]))
        with np.load(OUT/'run293/source_64_alias_map.npz') as z:alias=z['fragment_representative']
        canonical_graph=CanonicalCapacityGraph(graph.profiles,graph.offsets,graph.spans);canonical_graph.alias=alias
        c0=solve_case('64',canonical_graph,compiled,target,dest);write(dest/'C0_capacity.json',c0)
        index={int(f):i for i,f in enumerate(case['canonical'])}
        for pi,parent in enumerate(compiled['parents']):
            alternatives=parent['children'][0]['alternatives'];best_probability=None;best_rows=None
            for alt in alternatives:
                try:frags=canonical_graph.witness_fragments(alt)
                except ValueError:continue
                for side in (0,1):
                    ids=[index[f]*2+side for f in frags];bottleneck=float(p[ids].min())
                    if best_probability is None or bottleneck>best_probability:best_probability=bottleneck;best_rows=ids
            rows[pi].update(C0='PROVEN_FEASIBLE' if c0['status']=='COMPLETE_JOINT_GEOMETRIC_CAPACITY_PASSED' else 'NOT_PROVEN',
                all_compatible_curve_alternatives=len(alternatives),best_full_witness_min_probability=best_probability,
                best_full_witness_object_rows=best_rows)
        save();check_resources()
        # C1 enumerates every contiguous positive run on every profile and side.
        # Its full interval represents all possible endpoint/subchain choices.
        phase('C1_ALL_POSITIVE_DOMAIN_PATHS')
        groups={}
        for row in np.flatnonzero(p>=.5):
            fragment=int(case['canonical'][row//2]);groups.setdefault((graph.profile(fragment),int(row%2)),[]).append(fragment)
        runs=[]
        for (pid,side),fragments in groups.items():
            blocks=[]
            for f in sorted(fragments):
                if not blocks or f!=blocks[-1][-1]+1:blocks.append([f])
                else:blocks[-1].append(f)
            for block in blocks:runs.append(dict(profile=pid,side=side,fragments=block,
                lo=float(graph.spans[block[0],0]),hi=float(graph.spans[block[-1],1])))
        c1options=[[] for _ in rows];c1_exact=all(graph.curve(r['profile']).kind=='line' for r in runs)
        for pi,parent in enumerate(target['parents']):
            for run in runs:
                curve=graph.curve(run['profile'])
                if curve.kind!='line':continue
                domain=dict(representation=curve.segment(run['lo'],run['hi']))
                event=compatible_interval(domain,parent['children'][0]['representation'])
                if event is None:continue
                alt=dict(profile=run['profile'],lo=run['lo']+event['lo'],hi=run['lo']+event['hi'],
                    forward=event['target_direction_forward'],closed=False,wrap=False)
                frags=graph.witness_fragments(alt)
                c1options[pi].append(dict(slots=[f'{f}:{run["side"]}' for f in frags],alternative=alt,side=run['side']))
        c1=joint_fixed_sides(c1options,dest,'C1') if c1_exact else dict(status='NOT_PROVEN',reason='Positive arc domain requires full arc proof')
        for i,options in enumerate(c1options):
            rows[i].update(C1=('PROVEN_FEASIBLE' if i in c1.get('selected_parent_indices',[]) else 'PROVEN_INFEASIBLE' if not options and c1_exact else 'NOT_PROVEN'),
                           C1_alternative_count=len(options))
        save();check_resources()
        phase('C2_EXHAUSTIVE_FIXED_PREDICTED_PATHS')
        # This DEV run has only three selected nodes. Enumerate every simple
        # same-child path; no top-k/beam search and no actual-output rewriting.
        if len(nodes)>12:raise RuntimeError('Exhaustive path diagnostic capacity exceeded; do not truncate')
        adjacency={i:[] for i in range(len(nodes))}
        for i,a in enumerate(nodes):
            for j,b in enumerate(nodes):
                if i!=j and pair_features(a,b,graph)[1][0]:adjacency[i].append(j)
        paths=[]
        def visit(path):
            paths.append(path)
            for j in adjacency[path[-1]]:
                if j not in path:visit(path+[j])
        for i in range(len(nodes)):visit([i])
        c2options=[[] for _ in rows];native=native_targets(target)
        refs=[reference_segment(t['children'][0]) for t in native['parents']]
        for path in paths:
            local=[nodes[i] for i in path]
            dec=decode_paths(local,dict(selected_nodes=list(range(len(path))),selected_links=[(i,i+1,1) for i in range(len(path)-1)]),graph,[])
            if dec['invalid_predictions'] or len(dec['parents'])!=1 or dec['parents'][0]['child_count']!=1:continue
            par=dec['parents'][0]
            for pi,truth in enumerate(native['parents']):
                for view in variants(par,True):
                    if child_error(view['segments'][0][1],truth['children'][0],refs[pi],False,False) is not None:
                        c2options[pi].append(dict(slots=[f'{n["fragment"]}:{n["side"]}' for n in local],object_rows=[n['object_row'] for n in local]));break
        c2=joint_fixed_sides(c2options,dest,'C2')
        for i,options in enumerate(c2options):
            rows[i]['C2']='PROVEN_FEASIBLE' if i in c2.get('selected_parent_indices',[]) else 'PROVEN_INFEASIBLE' if not options else 'NOT_PROVEN'
            rows[i]['C2_alternative_count']=len(options)
            stage=next((s for s in ('C0','C1','C2') if rows[i][s]!='PROVEN_FEASIBLE'),'ACTUAL_ASSEMBLY')
            rows[i].update(failure_stage=stage,reason={'C0':'Full representation not proved','C1':'All threshold-positive finite domains exclude this complete target',
                'C2':'No fixed predicted geometry and category path passes D3','ACTUAL_ASSEMBLY':'Conditional geometry feasible; compare actual selection'}[stage])
        save()
        from pathlib import Path
        assert all(digest(Path(path))==sha for path,sha in immutable.items())
        write(dest/'worker_receipt.json',dict(status='DEV_FROZEN_COVERAGE_DIAGNOSTIC_COMPLETE',source='64',G=26,
            object_rows=n,positive_objects=len(nodes),C0=c0['status'],C1=c1,C2=c2,parent_rows=rows,
            full_parent_coverage_counts={s:sum(r[s]=='PROVEN_FEASIBLE' for r in rows) for s in ('C0','C1','C2')},
            positive_domain_runs=len(runs),exhaustive_predicted_paths=len(paths),
            model_and_actual_predictions_unchanged=True,DEV_used_for_training=False,new_fit_calls=0,
            not_extraction_accuracy=True,actual_latest_TP=0,actual_latest_FP=3,actual_latest_FN=26,
            seconds=time.perf_counter()-start,utc=utc()))
        phase('COMPLETE')
    except BaseException:
        save();write(dest/'worker_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,new_fit_calls=0,utc=utc()));raise


if __name__=='__main__':main()
