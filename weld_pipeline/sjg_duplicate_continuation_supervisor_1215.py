"""Pro374: frozen best367 plus trainable pair-set residual, final weights only."""
import argparse,ast,hashlib,os,shutil,time,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,resources,utc
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    started=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);completed=[]
    pro=read(OUT/'pro_review_run374/receipt.json')
    assert pro['assistant_sentinel_verified'] and pro['generation_stopped']
    assert hashlib.sha256((OUT/'pro_review_run374/answer.md').read_text(encoding='utf-8').encode()).hexdigest()==pro['answer_sha256']
    assert digest(OUT/'pro_review_run374/packet.md')==pro['packet_sha256']
    budget=450.;deadline=started+budget
    for number in (1159,1162,1173,1174,1178,1181,1184,1185,1186,1189,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215):
        for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    write(dest/'registered_protocol.json',dict(status='PRO374_RESUME374_HEAD112_TO168' ,
        execution_budget_seconds=budget,old_unused_balance_seconds=378.267712400004,old_balance_not_added=True,
        forest_fit_calls_allowed=0,base_Adam_step=560,base_parameters_moments_normalization_frozen=True,
        initial_branch_Adam_step=112,selector_updates=168,new_updates_required=56,trainable_parameters=145,frozen_residual_Adam_step=112,frozen_residual_run='run373',
        selector_stage_epochs=2,seven_pieces_per_epoch=True,batches_per_piece=4,clean_weight=.5,recovery_weight=.5,
        fresh_complete_trajectories_per_visit=2,trajectory_frozen_between_four_batches=True,
        pair_encoder_shape=[16,8,1],pair_encoder_activation='tanh',symmetric_forward_reverse_average=True,all_selected_scalar_max=True,
        frozen_residual_head_shape=[178,16,1],nonnegative_learned_pair_deduction=True,STOP_frozen_at_current_state=True,
        zero_output_layer_initialization=False,full_run374_state_resumed=True,independent_PCG64_seed=1031,initialization_std=.02,
        learned_encoding_cache_invalidated_after_each_update=True,true_pair_encoder_backpropagation=True,
        resident_trajectory_feature_tensor_allowed=False,streaming_one_source_one_trajectory_one_prefix=True,
        loss_kind='pairwise_plus_coverage_plus_pair_duplicate',coverage_coefficient=1.,duplicate_coefficient=1.,margin=0,temperature=1,
        pair_label='2-minus-two-parent-maximum-matching',unknown_pairs_auxiliary_mask_only=True,
        exact_opposite_label_input_conflict_stop=True,run374_initial_TP=206,run374_initial_FP=42,
        supervision_groups=['augmenting','covered_compatible','incompatible'],all_available_supervised_candidates=True,
        branch_moments_initially_zero=False,learning_rate=.001,weight_decay=.0001,clip=5.,
        original_TRAIN_sources=['21','4','10','25','36','42','BM79A_native_local'],TRAIN_G=363,pilot_G=500,
        all_candidates_normalization_and_inference_argmax_frozen=True,
        original_relation_feature='max_mean32_and_nearest_complete_parent16',R_dimension=48,
        nearest_tie_break='distance_then_frozen_candidate_index',encoded_max_tie='lowest_frozen_candidate_index',
        baseline='run367',baseline_TP=198,baseline_FP=41,baseline_preserved=True,missing_GT_not_removed=True,
        controls=['full112_state_and_same_input_forward_restore','112_clip_factors_and_discarded_Adam_copy_equivalence','frozen374_actual_errors_before_after'],
        final_test_order=['21','BM79A_native_local','10','64'],formal_raw_PCD_time_seconds=10,
        path_development_seconds=60,selection_development_seconds=120,full_contract_pass=False,utc=utc()))
    checks=[]
    for i in range(3):
        if i:time.sleep(1)
        r=resources();r['disk_free_bytes']=shutil.disk_usage(ROOT).free
        r['passed']=r['available_physical_bytes']>=3758096384 and r['remaining_commit_bytes']>=5368709120 and r['disk_free_bytes']>=5637144576
        checks.append(r);print('STARTUP_HEADROOM',i+1,r,flush=True)
    write(dest/'startup_headroom.json',dict(checks=checks,passed=all(v['passed'] for v in checks)))
    if not all(v['passed'] for v in checks):
        write(dest/'receipt.json',dict(status='NO_START_RESOURCE_HEADROOM',updates=0,seconds=time.perf_counter()-started,full_contract_pass=False));return
    with (OUT/'single_B_trial_Pro374.json').open('x',encoding='utf-8') as stream:stream.write(args.run+'\n'+utc())
    def stage(name,key='',limit=None):
        remaining=deadline-time.perf_counter()
        if remaining<=0:raise TimeoutError('Shared Pro374 B trial balance exhausted')
        if limit is not None:remaining=min(remaining,limit)
        if name in ('duplicate_control','duplicate_train','select'):module='weld_pipeline.sjg_duplicate_continuation_worker_1214'
        elif name in ('aggregate','score'):module='weld_pipeline.sjg_selector_trial_worker_1174'
        elif name in ('graph','solve','decode','test_graph','test_solve','test_decode'):module='weld_pipeline.sjg_sparse_recovery_worker_1170'
        elif name=='screen':module='weld_pipeline.sjg_recall_recovery_worker_1165'
        else:module='weld_pipeline.sjg_recall_set_worker_1162'
        directory=dest/f'job_{name}_{key}';directory.mkdir(exist_ok=False)
        command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',module,'--run',args.run,'--stage',name,'--source',key]
        if name in ('duplicate_control','duplicate_train','select'):command.extend(['--limit-seconds',str(remaining)])
        print('B_TRIAL_STAGE',name,key,'SECONDS_AVAILABLE',remaining,flush=True)
        result=supervise(command,directory,remaining)
        if result['termination_reason'] or result['exit_code']!=0:
            write(dest/'failure.json',dict(status='B_TRIAL_STAGE_FAILED',stage=name,source=key,watchdog=result,
                completed_stages=completed,seconds=time.perf_counter()-started,execution_budget_seconds=budget,
                automatic_retry=False,training_status='TRAIN_COMPLETE_EVALUATION_INCOMPLETE' if (dest/'selector_training.json').exists() else 'TRAIN_INCOMPLETE',full_contract_pass=False))
            raise RuntimeError('B trial stopped; preserve complete artifacts, no automatic extension')
        assert read(dest/f'stage_{name}_{key}_complete.json')['status']=='STAGE_COMPLETE'
        completed.append(dict(stage=name,source=key));write(dest/'progress.json',dict(completed_stages=completed,seconds=time.perf_counter()-started))
    try:
        old=OUT/'run367';binding=read(old/'upstream_verified.json');assert digest(old/'model.pkl')==binding['model_sha256']
        os.link(old/'model.pkl',dest/'model.pkl');shutil.copy2(old/'upstream_verified.json',dest/'upstream_verified.json')
        reused={}
        for key in ('21','4','10','25','36','42','BM79A_native_local'):
            source=old/f'source_{key}';target=dest/f'source_{key}';target.mkdir()
            r=read(source/'receipt.json');c=read(source/'compatibility.json')
            assert r['upstream_model_sha256']==binding['model_sha256']
            assert digest(source/'decoded.json')==r['decoded_sha256']==c['prediction_sha256']
            assert digest(source/'unary.npy')==r['unary_sha256']
            assert read(old/f'stage_match_{key}_complete.json')['status']=='STAGE_COMPLETE'
            files={}
            for path in source.iterdir():
                if path.is_file():os.link(path,target/path.name);files[path.name]=digest(path)
            os.link(old/f'screen_{key}.json',dest/f'screen_{key}.json')
            shutil.copy2(old/f'stage_match_{key}_complete.json',dest/f'stage_match_{key}_complete.json')
            reused[key]=files
        write(dest/'frozen_sources_reused.json',reused)
        shutil.copy2(old/'all_train_candidate_coverage.json',dest/'all_train_candidate_coverage.json')
        snapshot=dest/'source_snapshot';snapshot.mkdir()
        for number in (1158,1159,1160,1161,1162,1164,1165,1168,1169,1170,1173,1174,1176,1178,1181,1184,1185,1186,1189,1192,1193,1194,1195,1196,1197,1198,1199,1200,1201,1202,1203,1204,1205,1206,1207,1208,1209,1210,1211,1212,1213,1214,1215):
            for path in (ROOT/'weld_pipeline').glob(f'*_{number}.py'):shutil.copy2(path,snapshot/path.name)
        stage('duplicate_control')
        stage('duplicate_train')
        trained=read(dest/'selector_training.json')
        assert trained['updates']==168 and trained['new_updates']==56 and trained['base_Adam_step']==560 and trained['base_parameters_and_moments_frozen'] and trained['frozen_residual_parameters_and_moments_frozen']
        assert digest(dest/'selector.npz')==trained['selector_sha256']
        for key in ('21','BM79A_native_local','10'):
            stage('test_upstream',key);stage('select',key,120.);stage('score',key)
        stage('aggregate')
        result=read(dest/'three_case_final_summary.json')
        if not result['geometric_continuation_pass']:
            write(dest/'receipt.json',dict(status='B_FINAL_THREE_CASE_GEOMETRY_FAILED',summary=result,
                selector_updates=168,new_updates=56,base_Adam_step=560,seconds=time.perf_counter()-started,full_contract_pass=False));return
        key='64';stage('screen',key);stage('test_graph',key);stage('test_solve',key,60.);stage('test_decode',key)
        stage('select',key,120.);stage('score',key)
        write(dest/'receipt.json',dict(status='B_PILOT_AND_DEV_TESTS_COMPLETED',summary=result,
            DEV_report=read(dest/'test_64/evaluation.json')['report'],selector_updates=168,new_updates=56,base_Adam_step=560,
            seconds=time.perf_counter()-started,full_contract_pass=False))
    except BaseException:
        if not (dest/'failure.json').exists():write(dest/'failure.json',dict(status='B_TRIAL_CONTROLLER_FAILED',error=traceback.format_exc(),
            completed_stages=completed,seconds=time.perf_counter()-started,automatic_retry=False,full_contract_pass=False))
        raise


if __name__=='__main__':main()
