"""Freeze every inventoried version into family-grouped train/validation/test.

Historical training and role-uncertain legacy families remain training-only.
Untouched expansion holdout families supply the independent test set. Labels
are qualified after assignment; eligibility never changes a family's split.
"""
from pathlib import Path
import collections
import hashlib
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/full_dataset_v1'
OLD=ROOT/'server_rebuild/expansion_v1'


def rank(family):
    return hashlib.sha256(('FULL_SPLIT_1031_'+family['family_id']).encode()).hexdigest()


def make_split(manifest, inventory, metadata):
    groups=manifest['groups'];bydir={r['directory']:r for r in inventory['cases']}
    audits={r['directory']:r for r in metadata['cases']}
    directories=[d for g in groups for d in g['directories']]
    if len(directories)!=len(set(directories)) or set(directories)!=set(bydir):
        raise ValueError('Family groups must cover every inventory directory exactly once')
    held=[g for g in groups if g['role']=='EXPANSION_DEV_VALID']
    if len(held)!=15:
        raise ValueError('Unexpected historical untouched holdout pool; inspect before changing assignment')
    details={}
    for group in groups:
        books=[w for d in group['directories'] for w in audits[d]['workbooks']]
        details[group['family_id']]=dict(curved=any(w['curved_parents'] for w in books),
                                       children=max(w['children'] for w in books))
    targets={'validation':7,'test':8};assignments={};counts=collections.Counter();children=collections.Counter()
    # Put one curve family in each holdout using existing global metadata only.
    curved=sorted([g for g in held if details[g['family_id']]['curved']],key=rank)
    if len(curved)<2:
        raise ValueError('Insufficient untouched curve families for both holdouts')
    for group,split in zip(curved[:2],['validation','test']):
        assignments[group['family_id']]=split;counts[split]+=1;children[split]+=details[group['family_id']]['children']
    remaining=sorted([g for g in held if g['family_id'] not in assignments],
                     key=lambda g:(-details[g['family_id']]['children'],rank(g)))
    for group in remaining:
        available=[s for s in targets if counts[s]<targets[s]]
        split=min(available,key=lambda s:(children[s]/targets[s],counts[s]/targets[s],s))
        assignments[group['family_id']]=split;counts[split]+=1;children[split]+=details[group['family_id']]['children']
    families=[];examples=[];seen_hashes={}
    for group in groups:
        role=group['role']
        split=assignments.get(group['family_id'],'validation' if role=='PROTECTED_ORIGINAL_DEV' else 'train')
        independent=role=='EXPANSION_DEV_VALID'
        reason='Untouched historical holdout, metadata-stratified before new fit' if independent else (
               'Protected original development family, diagnostic validation only' if role=='PROTECTED_ORIGINAL_DEV' else
               'Historical training exposure or unresolved legacy exposure: training-only')
        families.append(dict(family_id=group['family_id'],key=group['key'],split=split,previous_role=role,
                             independent_holdout=independent,assignment_reason=reason,directories=group['directories'],
                             pcd_sha256=group['pcd_sha256'],metadata=details[group['family_id']]))
        for sha in group['pcd_sha256']:
            if sha in seen_hashes and seen_hashes[sha]!=group['family_id']:
                raise ValueError('Identical raw cloud appears in different families')
            seen_hashes[sha]=group['family_id']
        preferred=group['representative_directory'] or min(group['directories'])
        for directory in group['directories']:
            example_key=group['key'] if directory==preferred else group['key']+'_v'+hashlib.sha256(directory.encode()).hexdigest()[:8]
            examples.append(dict(key=example_key,family_key=group['key'],family_id=group['family_id'],split=split,
                                 independent_holdout=independent,directory=directory,
                                 canonical_version=directory==preferred,source_inventory=bydir[directory],
                                 eligibility='PENDING_SOURCE_AND_LABEL_QUALIFICATION'))
    return dict(schema='FULL_FAMILY_DATASET_SPLIT_V1',seed=1031,families=families,examples=examples,
                family_counts=dict(collections.Counter(g['split'] for g in families)),
                directory_counts=dict(collections.Counter(x['split'] for x in examples)),
                untouched_holdout_family_counts=dict(counts),every_inventory_directory_assigned=True,
                no_family_crosses_split=True,raw_cloud_byte_duplicates_do_not_cross_families=True,
                historical_64_training_forbidden=True,test_for_final_evaluation_only=True,
                validation_for_selection_only=True,all_versions_utilized_within_assigned_role=True,
                qualification_does_not_move_examples_between_splits=True,
                residual_derivative_grouping_risk=manifest['residual_unknown_cropped_derivative_leakage_risk'])


def main():
    sources=[OLD/'group_manifest.json',ROOT/'migration_checks/data_inventory.json',ROOT/'migration_checks/dataset_audit.json']
    value=make_split(*[read(p) for p in sources])
    value['source_bindings']={str(p.relative_to(ROOT)):digest(p) for p in sources}
    value['source_sha256']=digest(__file__)
    BASE.mkdir(exist_ok=False)
    write(BASE/'split_manifest.json',value)
    for split in ['train','validation','test']:
        write(BASE/(split+'.json'),dict(split=split,split_manifest_sha256=digest(BASE/'split_manifest.json'),
              families=[x for x in value['families'] if x['split']==split],
              examples=[x for x in value['examples'] if x['split']==split]))
    write(BASE/'split_validation.json',dict(status='PASS',families=len(value['families']),directories=len(value['examples']),
          family_counts=value['family_counts'],directory_counts=value['directory_counts'],
          independent_test_families=sum(x['split']=='test' and x['independent_holdout'] for x in value['families']),
          all_directories_assigned_exactly_once=True,no_cross_split_families=True,
          original_dev_training_excluded=True,split_manifest_sha256=digest(BASE/'split_manifest.json'),fit_calls=0))
    print('FULL_SPLIT_FROZEN',value['family_counts'],value['directory_counts'],flush=True)


if __name__=='__main__':main()
