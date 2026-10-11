"""Freeze geometry query bank before measuring TRAIN-only diagnostic cost."""
import argparse
import gc
import shutil
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_geometry_query_bank_1025 import build_bank,selected_cost,controls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);(dest/'source_snapshot').mkdir()
    for name in ('sjg_geometry_query_bank_1025.py','sjg_geometry_bank_audit_1026.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,dest/'source_snapshot'/name)
    write(dest/'registered_protocol.json',dict(stage='XYZ_QUERY_SHARING_NOT_SEMANTIC_MERGE',source='run201',
        TRAIN_G=363,all_G=500,all_banks_frozen_before_TRAIN_labels=True,optimizer_updates=0,start_utc=utc()))
    try:
        write(dest/'controls.json',controls());rows=[]
        for source in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases']:
            start=time.perf_counter();check_resources();key=source['source_key'];path=OUT/'run201'/source['json_file']
            assert digest(path)==source['json_sha256'];domain=read(path);bank,mapping=build_bank(domain)
            jp=dest/f'source_{key}_geometry_bank.json';mp=dest/f'source_{key}_family_to_bank.npy'
            bank['source_joint_domain_sha256']=source['json_sha256'];write(jp,bank);np.save(mp,mapping)
            row=dict(source_key=key,json_file=jp.name,json_sha256=digest(jp),mapping_file=mp.name,mapping_sha256=digest(mp),
                semantic_families=bank['semantic_family_count'],query_geometries=bank['query_geometry_count'],
                length_compression=bank['duplicated_XYZ_length_compression'],construction_seconds=time.perf_counter()-start)
            rows.append(row);print('XYZ_SHARED_BANK',key,row['semantic_families'],'to',row['query_geometries'],
                'length_ratio',round(row['length_compression'],3),'semantic_merges',0,flush=True)
            del domain,bank,mapping;gc.collect()
        write(dest/'all_banks_frozen_before_labels.json',dict(cases=rows,target_reads=0))
        audits=[]
        for case in read(OUT/'run186/training_manifest.json')['cases']:
            key=case['source_key'];row=next(r for r in rows if r['source_key']==key)
            assert digest(dest/row['json_file'])==row['json_sha256']
            assert digest(dest/row['mapping_file'])==row['mapping_sha256']
            domain=read(OUT/f'run201/source_{key}_joint_domain.json');bank=read(dest/row['json_file'])
            mapping=np.load(dest/row['mapping_file'],allow_pickle=False)
            with np.load(OUT/f'run201/source_{key}_joint_mapping.npz',allow_pickle=False) as z:fm=z['profile_family']
            supervision=read(OUT/f'run186/source_{key}_supervision.json')
            compatible=sorted(set(int(fm[int(p)]) for p in supervision['by_profile']))
            cost=selected_cost(domain,bank,mapping,compatible);cost['source_key']=key;audits.append(cost)
            print('TRAIN_COMPATIBLE_QUERY_COST',key,cost,flush=True)
        write(dest/'TRAIN_compute_diagnostic.json',dict(cases=audits,oracle_compatibility_not_prediction=True,
            no_runtime_GT_filter=True,semantic_scoring_cost_not_included=True,optimizer_updates=0))
        write(dest/'receipt.json',dict(status='GEOMETRY_QUERY_BANK_FROZEN',TRAIN_G=363,all_G=500,
            semantic_merges=0,no_model_accuracy_claim=True,selected_runtime_budgets_not_tested=True,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),utc=utc()));raise


if __name__=='__main__':main()
