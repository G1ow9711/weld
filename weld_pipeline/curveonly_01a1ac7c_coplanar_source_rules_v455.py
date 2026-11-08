"""Use the frozen source expert on a verified coplanar two-panel interface.

A joint subtype is not a no-weld rule. Endorsed sources and confirmed model
common-plane anchors stay eligible; an expert abstention is logged separately
from a negative class. Geometry and units of retained parents are unchanged.
"""
import copy,time
from . import curveonly_01a1ac7c_coplanar_observations_v453 as observation


def select(decoded,observed):
    votes={int(v['candidate']):v for v in decoded.get('paired_face_source_rule',{}).get('source_votes',[])}
    protected=observation.protected_sources(decoded);chosen=[];retained=[]
    for row in observed.get('qualifications',[]):
        q=row['geometry'];index=row['parent_index'];parent=decoded['parents'][index];ci=parent['source_candidate']
        if ci!=row['candidate']:raise ValueError('Changed parent source invalidates finite role observation')
        if not(q.get('passed') and q.get('verified_dense')):continue
        interval=[parent['interval']['start_mm'],parent['interval']['end_mm']]
        if interval!=q['source_interval_mm']:raise ValueError('Changed native interval invalidates finite role observation')
        vote=votes.get(ci)
        if vote is None or vote['endorsed'] or ci in protected:
            retained.append(dict(parent_index=index,candidate=ci,reason='POSITIVE_OR_UNKNOWN_SOURCE_OR_CONFIRMED_MODEL_STEP_ANCHOR'));continue
        chosen.append(dict(parent_index=index,candidate=ci,finite_interface=q,source_vote=vote,
            decision='FROZEN_SOURCE_EXPERT_NEGATIVE' if vote['rejected'] else 'FROZEN_SOURCE_EXPERT_ABSTAIN',
            joint_geometry_not_treated_as_no_weld_truth=True))
    return chosen,dict(branch='FINITE_COPLANAR_PANEL_SOURCE_ROUTE_V455',certificates=chosen,retained=retained,
        input_geometry_only_subtype=True,model_source_threshold_unchanged=True,no_fixed_thickness_radius_name_or_count_switch=True,
        unknown_or_unqualified_geometry_preserves_current_model=True,no_new_fit_or_VALcalibration=True)


def correct(decoded,features,packet,inventory,solids,cached_observation=None):
    started=time.perf_counter()
    observed=observation.observe(decoded,packet,inventory,solids,verify=True) if cached_observation is None else copy.deepcopy(cached_observation)
    chosen,audit=select(decoded,observed);result=copy.deepcopy(decoded)
    remove={q['parent_index'] for q in chosen}
    if remove:
        parents=[p for i,p in enumerate(result['parents']) if i not in remove]
        for i,p in enumerate(parents):p['parent_id']=f'curve_{i:05d}'
        result.update(parents=parents,geometry_units=len(parents),published_segments=sum(len(p['segments']) for p in parents))
    audit.update(replaced_upstream_parent_indices=sorted(remove),new_parent_count=0,
        exact_bound_finite_role_cache_used=cached_observation is not None,
        observation_reason=observed.get('reason'),untargeted_parent_fields_retained_exact_except_parent_id=True,seconds=time.perf_counter()-started)
    result['coplanar_panel_source_rule']=audit
    return result
