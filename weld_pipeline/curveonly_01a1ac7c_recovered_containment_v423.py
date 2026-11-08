"""Remove contained same-source representations after certified chain recovery."""
import copy
from . import curveonly_01a1ac7c_native_join_rules_v418 as recovery


def correct(decoded):
    result=copy.deepcopy(decoded);audit=decoded.get('wall_base_chain_rule',{})
    certificates={c['candidate']:c for c in audit.get('certificates',[]) if
        c.get('verified_dense') and c.get('existing_model_native_join_intent',{}).get('anchors')}
    n=int(audit.get('new_parent_count',0));parents=decoded['parents']
    evidence=[]
    if certificates and 0<n<=len(parents):
        first_new=len(parents)-n
        for i,parent in enumerate(parents[:first_new]):
            ci=parent['source_candidate'];r=parent['interval']
            if ci not in certificates:continue
            for j in range(first_new,len(parents)):
                other=parents[j];s=other['interval']
                if other['source_candidate']!=ci or parent['weld_type']!=other['weld_type']:continue
                if parent.get('weld_side')!=other.get('weld_side'):continue
                eps=recovery.old.step.NUMERIC
                if not(s['start_mm']<=r['start_mm']+eps and s['end_mm']>=r['end_mm']-eps):continue
                if s['end_mm']-s['start_mm']<=r['end_mm']-r['start_mm']+eps:continue
                projection=recovery.old.aliases.coincident_parameters(parent['points'],other['points'])
                if projection is None:continue
                _,error=projection
                evidence.append(dict(removed_parent_index=i,covering_new_parent_index=j,candidate=ci,
                    contained_native_interval=r,covering_native_interval=s,
                    published_ordered_projection_error_mm=float(error),
                    reason='SAME_SOURCE_SIDE_TYPE_AND_FULL_NATIVE_CONTAINMENT_AFTER_VERIFIED_RECOVERY'))
                break
    removed={v['removed_parent_index'] for v in evidence}
    if removed:
        result['parents']=[p for i,p in enumerate(result['parents']) if i not in removed]
        for i,p in enumerate(result['parents']):p['parent_id']=f'curve_{i:05d}'
        result['geometry_units']=len(result['parents'])
        result['published_segments']=sum(len(p['segments']) for p in result['parents'])
    result['recovered_native_containment_rule']=dict(branch='V423_CERTIFIED_RECOVERY_NATIVE_CONTAINMENT',
        certificates=evidence,removed_parent_indices=sorted(removed),
        no_short_length_filter=True,no_new_fit_or_VALcalibration=True,
        original_parent_fields_retained_except_reindexed_parent_id=True)
    return result
