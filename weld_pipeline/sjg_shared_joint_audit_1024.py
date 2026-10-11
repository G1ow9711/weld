"""Experiment 1: freeze all pure-input joint maps, then audit TRAIN witnesses."""
import argparse
import gc
import shutil
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_shared_joint_domain_1023 import build,controls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    snapshots=dest/'source_snapshot';snapshots.mkdir()
    for name in ('sjg_shared_joint_domain_1023.py','sjg_shared_joint_audit_1024.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snapshots/name)
    protocol=dict(stage='SHARED_JOINT_DOMAIN_ZERO_TRAINING',start_utc=utc(),
        consultation='pro_review_run200/receipt.json',all_nine_G=500,TRAIN_G=363,
        model_optimizer_updates=0,coarse_relation_token_budget=100000,fine_query_runtime_budget=100000,
        XYZ_profiles_source='run185',existing_graph_source='run186',
        support_and_gap_semantics='Every original profile remains authoritative; no candidate removed',
        unknown_material_relations_kept=True,side_and_ordered_face_associations_preserved=True,
        hard_exclusivity=False,all_input_maps_frozen_before_TRAIN_supervision=True,
        full_contract_pass=False)
    write(dest/'registered_protocol.json',protocol)
    try:
        assert read(OUT/'pro_review_run200/receipt.json')['generation_stopped']
        write(dest/'controls.json',controls())
        cases=read(OUT/'run98/local_cohort.json')['cases'];frozen=[]
        original_graphs={c['source_key']:c for c in read(OUT/'run186/all_graphs_frozen_before_labels.json')['cases']}
        for case in cases:
            start=time.perf_counter();check_resources();key=case['source_key']
            folder=OUT/f'run185/source_{key}';binding=read(folder/'input_freeze.json')
            assert binding['input_sha256']==case['input_sha256']
            assert digest(folder/'profiles.json')==binding['files']['profiles.json']
            assert digest(folder/'surfaces.json')==binding['files']['surfaces.json']
            assert digest(OUT/'run186'/original_graphs[key]['file'])==original_graphs[key]['sha256']
            profiles=read(folder/'profiles.json');graph,arrays=build(profiles)
            assert len(profiles)==original_graphs[key]['profiles']
            graph['input_binding']=dict(profile_file=str((folder/'profiles.json').relative_to(OUT)),
                 profile_sha256=binding['files']['profiles.json'],face_file=str((folder/'surfaces.json').relative_to(OUT)),
                 face_sha256=binding['files']['surfaces.json'],source_XYZ_sha256=case['input_sha256'],
                 original_graph=original_graphs[key],material_relation_is_predicted_not_truth=True)
            jp=dest/f'source_{key}_joint_domain.json';npz=dest/f'source_{key}_joint_mapping.npz'
            write(jp,graph);np.savez(npz,**arrays)
            row=dict(source_key=key,json_file=jp.name,json_sha256=digest(jp),mapping_file=npz.name,
                     mapping_sha256=digest(npz),stats=graph['stats'],construction_seconds=time.perf_counter()-start)
            frozen.append(row);write(dest/'progress.json',dict(stage='INPUT_DOMAIN_FROZEN',cases=frozen,utc=utc()))
            print('JOINT_DOMAIN',key,'profiles',len(profiles),'relations',graph['stats']['relations'],
                  'families',graph['stats']['geometry_families'],'share_ratio',round(graph['stats']['duplicate_length_compression'],3),
                  'seconds',round(row['construction_seconds'],3),flush=True)
            del profiles,graph,arrays;gc.collect()
        write(dest/'all_domains_frozen_before_labels.json',dict(cases=frozen,target_reads=0))
        manifest=read(OUT/'run186/training_manifest.json');audits=[]
        for old in manifest['cases']:
            key=old['source_key'];row=next(r for r in frozen if r['source_key']==key)
            assert digest(dest/row['json_file'])==row['json_sha256']
            assert digest(dest/row['mapping_file'])==row['mapping_sha256']
            graph=read(dest/row['json_file'])
            with np.load(dest/row['mapping_file'],allow_pickle=False) as z: family=z['profile_family'];relation=z['profile_relation']
            sup=read(OUT/f'run186/source_{key}_supervision.json');witnesses=[]
            assert sup['original_G']==old['original_G'] and not sup['missing']
            for child in sup['children']:
                ids=child['profiles'];assert ids
                for i in ids:
                    assert i in graph['families'][int(family[i])]['profiles']
                    assert i in graph['relations'][int(relation[i])]['profiles']
                    entries=sup['by_profile'][str(i)]
                    assert any(e['original_child_id']==child['original_child_id'] for e in entries)
                witnesses.append(dict(original_child_id=child['original_child_id'],kind=child['kind'],
                    original_compatible_profiles=len(ids),retained_compatible_profiles=len(ids),
                    relation_ids=sorted(set(int(relation[i]) for i in ids))))
            audit=dict(source_key=key,G=sup['original_G'],retained_children=len(witnesses),
                 kind_counts=old['native_kind_counts'],witnesses=witnesses,
                 geometry_and_finite_domains_unchanged=True,simultaneous_children_not_excluded=True,
                 no_model_accuracy_claim=True)
            audits.append(audit)
            print('TRAIN_DOMAIN_RETAINED',key,len(witnesses),'/',sup['original_G'],flush=True)
        total=sum(a['G'] for a in audits);retained=sum(a['retained_children'] for a in audits)
        assert total==retained==363
        assert sum(a['kind_counts']['arc'] for a in audits)==3
        relation_pass=all(r['stats']['relation_input_budget_pass'] for r in frozen)
        write(dest/'TRAIN_domain_retention.json',dict(cases=audits,G=total,retained=retained,
            existing_supervision_is_compatibility_not_material_truth=True,heldout_labels_not_read=True))
        write(dest/'receipt.json',dict(status='SHARED_DOMAIN_AND_RELATION_BUDGET_PASS' if relation_pass else 'RELATION_BUDGET_FAIL',
            TRAIN_G=363,retained_TRAIN=363,original_all_G=500,coarse_relation_budget_pass=relation_pass,
            selected_fine_query_budget_not_yet_evaluated=True,path_decoder_not_yet_implemented=True,
            full_contract_pass=False,optimizer_updates=0,end_utc=utc()))
        print('EXPERIMENT_1_INPUT_RETENTION',retained,'/',total,'RELATION_BUDGET',relation_pass,
              'NEW_MODEL_NOT_TRAINED',flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),utc=utc()));raise


if __name__=='__main__':main()
