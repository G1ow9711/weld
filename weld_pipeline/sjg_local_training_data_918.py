"""Freeze local graph inputs, then construct TRAIN-only line/arc supervision."""
import argparse
import gc
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_local_graph_data_916 import graph_inputs,queries,sample_training_queries,CanonicalQueryTree
from weld_pipeline.sjg_local_curve_supervision_915 import supervise,controls

def load_case(case,proposals,features):
    key=case['source_key'];folder=proposals/f'source_{key}';binding=read(folder/'input_freeze.json')
    fb=read(features/f'source_{key}_freeze.json')
    assert binding['input_sha256']==case['input_sha256']==fb['input_sha256']
    assert digest(OUT/case['input_file'])==case['input_sha256']
    assert digest(features/fb['feature_file'])==fb['feature_sha256']
    for file in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(folder/file)==binding['files'][file]
    xyz=np.load(OUT/case['input_file'],mmap_mode='r',allow_pickle=False)
    with np.load(features/fb['feature_file'],allow_pickle=False) as z:point={k:z[k] for k in ('embedding','normal','boundary')}
    return xyz,point,read(folder/'surfaces.json'),read(folder/'profiles.json'),folder

def main(default_proposal_run='run115',default_feature_run='run106',default_material_run=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--proposal-run',default=default_proposal_run)
    ap.add_argument('--feature-run',default=default_feature_run);ap.add_argument('--material-instances-run',default=default_material_run);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);proposals=OUT/args.proposal_run;features=OUT/args.feature_run
    assert read(proposals/'receipt.json')['status'] in ('LOCAL_RANK2_TYPED_CURVES_FROZEN','UNIFIED_LOCAL_CANDIDATES_FROZEN','NATIVE_JOINT_NINE_CANDIDATES_FROZEN')
    assert read(OUT/'run99/receipt.json')['bm79a_rigid_copies_within_3mm_excluded_against_heldout']
    cohort=read(OUT/'run98/local_cohort.json');cases=cohort['cases'];graphs=[];training=[]
    train=[c['source_key'] for c in cases if c['role']=='TRAIN']+['BM79A_native_local']
    protocol=dict(stage='LOCAL_GRAPH_AND_TYPED_TRAINING_QUERIES',start_utc=utc(),source_sha256=digest(__file__),
        source_run=args.proposal_run,features=args.feature_run,material_instances_run=args.material_instances_run,train=train,development=['64'],arc_diagnostic_not_train=['24'],
        original_eight_G=496,new_BM79A_TRAIN_G=4,training_G=363,queries_per_case=2048,seed=918,
        native_local_XYZ_position_is_input=True,source_identity_not_input=True,STEP_not_runtime_input=True,
        query_neighbors='1mm_global_representatives_mapped_to_original_feature_indices',
        query_source_sha256=digest(ROOT/'weld_pipeline/sjg_local_graph_data_916.py'),
        geometry_input_schema_changed_old_event_checkpoint_not_compatible=True,
        all_input_graphs_before_any_target_read=True,training_queries_use_TRAIN_labels=True,
        inference_must_use_free_input_carrier_grids=True,formal_independence_not_claimed=True,
        material_identity_and_E16_truth_not_fabricated=True,optimizer_updates=0,full_contract_pass=False)
    write(dest/'registered_protocol.json',protocol)
    try:
        write(dest/'typed_event_controls.json',controls())
        for case in cases:
            check_resources();key=case['source_key'];xyz,point,faces,profiles,folder=load_case(case,proposals,features)
            with np.load(folder/'surface_support.npz',allow_pickle=False) as support:
                graph=graph_inputs(xyz,point,faces,support,profiles)
                if args.material_instances_run:
                    from weld_pipeline.sjg_predicted_material_graph_984 import material_graph
                    binding=read(folder/'input_freeze.json')
                    instance_path=OUT/binding['instance_file']
                    assert instance_path.parent==OUT/args.material_instances_run
                    assert digest(instance_path)==binding['instance_sha256']
                    with np.load(instance_path,allow_pickle=False) as z:material=z['material_instance']
                    extra,metadata=material_graph(xyz,point['normal'],point['boundary'],material,[support[f['point_ids_key']] for f in faces])
                    graph.update(extra);write(dest/f'source_{key}_material_graph.json',metadata)
                    del material,extra
            file=dest/f'source_{key}_graph.npz';np.savez(file,**graph)
            row=dict(source_key=key,role='TRAIN' if key in train else case['role'],file=file.name,sha256=digest(file),
                     profiles=len(profiles),faces=len(faces),input_sha256=case['input_sha256'])
            graphs.append(row);print('LOCAL_GRAPH_FROZEN',key,len(faces),len(profiles),flush=True)
            del xyz,point,faces,profiles,graph;gc.collect()
        write(dest/'all_graphs_frozen_before_labels.json',dict(cases=graphs,target_reads=0))
        for index,case in enumerate(cases):
            key=case['source_key']
            if key not in train:continue
            check_resources();xyz,point,faces,profiles,folder=load_case(case,proposals,features)
            target_file=OUT/case['target_file'];assert digest(target_file)==case['target_sha256'];target=read(target_file)
            supervision=supervise(profiles,target);write(dest/f'source_{key}_supervision.json',supervision)
            sample=sample_training_queries(profiles,supervision,918+index,count=2048)
            patch,query=queries(profiles,sample['nodes'],sample['q_mm'],xyz,point,CanonicalQueryTree(xyz))
            inputs=dest/f'source_{key}_training_inputs.npz';labels=dest/f'source_{key}_training_labels.npz'
            np.savez(inputs,nodes=sample['nodes'],q_mm=sample['q_mm'],patch=patch,query=query)
            np.savez(labels,**{k:sample[k] for k in ('target','target_mask','offset','offset_mask')})
            row=dict(source_key=key,original_G=supervision['original_G'],covered_G=supervision['covered_children'],
                missing_G=len(supervision['missing']),query_count=len(sample['nodes']),
                input_file=inputs.name,input_sha256=digest(inputs),label_file=labels.name,label_sha256=digest(labels),
                target_sha256=case['target_sha256'],native_kind_counts={kind:sum(c['representation']['kind']==kind for p in target['parents'] for c in p['children']) for kind in ('line','arc')},
                covered_kind_counts={kind:sum(c['kind']==kind for c in supervision['children']) for kind in ('line','arc')})
            training.append(row);write(dest/'training_manifest.json',dict(cases=training,train=train))
            print('LOCAL_TYPED_TRAIN_DATA',key,'COVERED',row['covered_G'],'/',row['original_G'],'QUERIES',row['query_count'],flush=True)
            del xyz,point,faces,profiles,patch,query,sample;gc.collect()
        assert sum(r['original_G'] for r in training)==363
        write(dest/'receipt.json',dict(status='LOCAL_TYPED_GRAPH_TRAINING_DATA_READY',training=training,
            training_G=363,original_eight_G=496,features=args.feature_run,material_instances_run=args.material_instances_run,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
