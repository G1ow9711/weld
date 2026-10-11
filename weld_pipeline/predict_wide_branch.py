"""Raw inference using hash-verified equivalent adaptive-thread features."""
import argparse,json,time
from pathlib import Path
import torch
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.typed_sequences import make_sequences,FEATURE_VERSION
from weld_pipeline.typed_contact_model import TypedContactNet,infer_probabilities,decode_typed
from weld_pipeline.model_registry_extended import model_from_checkpoint
from weld_pipeline.surface_features import append_surface_features,FEATURE_VERSION as SURFACE_FEATURE_VERSION
from weld_pipeline.bidirectional_surface_features import append_bidirectional_surface_features,FEATURE_VERSION as BOTH_FEATURE_VERSION
from weld_pipeline.robust_sequences import make_sequences as make_robust_sequences
from weld_pipeline.bounded_sequences import make_sequences as make_bounded_sequences
from weld_pipeline.export import save_csv,save_all
from weld_pipeline.frame_context_features import append_frame_context,FEATURE_VERSION as CONTEXT_VERSION
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.wide_vertical_features import FEATURE_VERSION as MIXED_VERSION, make_vertical_context, ContextFeatures
from weld_pipeline.curved_sequences import decode_curved


def predict(pcd,checkpoint,outdir,with_excel=False,fast_mixed=False,decoder_config_path=None):
    torch.set_num_threads(4)
    wide_equivalence_path=Path('weld_pipeline/reports/wide_raw_feature_equivalence/receipt.json')
    wide_equivalence=json.loads(wide_equivalence_path.read_text(encoding='utf-8'))
    if wide_equivalence['status']!='PASS':raise ValueError('Wide raw feature equivalence missing')
    for source,wanted in wide_equivalence['sources'].items():
        if sha256(source)!=wanted:raise ValueError('Wide equivalence source changed: '+source)
    with Path(checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    surface=state.get('model_name') in ['surface_process','local_surface_process','wide_branch_surface','context_surface_process','arc_adapter_surface_process','straight_residual_surface_process']
    expected_version=SURFACE_FEATURE_VERSION if surface else FEATURE_VERSION
    if surface and state['feature_version']==BOTH_FEATURE_VERSION:expected_version=BOTH_FEATURE_VERSION
    if surface and state['feature_version']=='typed_contact_surface_v3':expected_version='typed_contact_surface_v3'
    if state.get('model_name')=='context_surface_process' and state['feature_version']==CONTEXT_VERSION:expected_version=CONTEXT_VERSION
    if surface and state['feature_version']==MIXED_VERSION:expected_version=MIXED_VERSION
    if state['feature_version']!=expected_version:raise ValueError('Checkpoint/feature version differs')
    interval_parameters=None;interval_sources=[]
    if decoder_config_path:
        from weld_pipeline.event_gap_config import load_interval_config
        interval_parameters,interval_config=load_interval_config(decoder_config_path,checkpoint)
        interval_sources=[Path('weld_pipeline/event_gap_config.py'),*[Path(p) for p in interval_config['sources']]]
    equivalence_path=None;extra_runtime_sources=[]
    if fast_mixed:
        if expected_version!=MIXED_VERSION:raise ValueError('Fast mixed generator requires mixed checkpoint')
        equivalence_path=Path('weld_pipeline/reports/pooled_shared_feature_equivalence/receipt.json')
        equivalence=json.loads(equivalence_path.read_text(encoding='utf-8'))
        if equivalence['status']!='PASS':raise ValueError('Fast feature equivalence has not passed')
        for source,wanted in equivalence['sources'].items():
            if sha256(source)!=wanted:raise ValueError('Fast feature source differs from verified version: '+source)
        extra_runtime_sources=[Path('weld_pipeline/pooled_shared_mixed_sequences.py'),Path('weld_pipeline/indexed_surface_features.py'),*[Path('weld_pipeline')/n for n in ['shared_arc_sequences.py','shared_tangent_sequences.py','shared_curved_contacts.py']]]
    source_names=['typed_contact_model.py','typed_sequences.py','surface_planes_local.py','surface_planes.py','geometry_candidates.py']
    if expected_version in ['typed_contact_surface_v3',CONTEXT_VERSION,MIXED_VERSION]:
        source_names.extend(['robust_sequences.py','robust_surface_planes.py','point_frame.py'])
    if state.get('bounded_neighbors'):
        if expected_version not in ['typed_contact_surface_v3',CONTEXT_VERSION,MIXED_VERSION]:raise ValueError('Bounded query version mismatch')
        source_names.append('bounded_sequences.py')
    if state.get('model_name')=='process_planes':source_names.append('process_plane_model.py')
    if surface:source_names.extend(['surface_process_model.py','surface_features.py','model_registry.py','process_plane_model.py'])
    if state.get('model_name') in ['local_surface_process','wide_branch_surface']:source_names.extend(['local_surface_model.py','model_registry_extended.py'])
    if state.get('model_name')=='wide_branch_surface':source_names.extend(['invariant_surface_model.py','model_registry_research.py'])
    if state.get('model_name') in ['arc_adapter_surface_process','straight_residual_surface_process']:source_names.extend(['local_surface_model.py','model_registry_extended.py',
        'invariant_surface_model.py','model_registry_research.py','arc_adapter_model.py','model_registry_adapter.py','model_registry_context.py'])
    if state.get('model_name')=='straight_residual_surface_process':
        source_names.extend(['straight_residual_model.py','model_registry_residual.py'])
    if state.get('model_name')=='context_surface_process':source_names.extend(['context_surface_model.py','frame_context_features.py','indexed_surface_features.py','local_height_features.py','model_registry_context.py','model_registry_research.py'])
    if state.get('fast_context'):
        if expected_version!=CONTEXT_VERSION:raise ValueError('Fast context feature mismatch')
        source_names.extend(['frame_context_features_fast.py','range_extrema.py'])
    if expected_version==BOTH_FEATURE_VERSION:source_names.append('bidirectional_surface_features.py')
    if expected_version==MIXED_VERSION:
        source_names.extend(['mixed_sequences.py','curved_sequences.py','arc_tangent_sequences.py','arc_surface_candidates.py',
                             'local_tangent_circles.py','tangent_circle_candidates.py','circle_candidates.py'])
    source_names.extend(['crossing_plane_model.py','model_registry_crossing.py','vertical_plane_model.py','model_registry_vertical.py','typed_branch_model.py','model_registry_branch.py'])
    source_names.extend(['wide_branch_model.py','model_registry_wide_branch.py','wide_vertical_features.py'])
    for name in source_names:
        path=Path('weld_pipeline')/name
        if state['source_hashes'].get(str(path))!=sha256(path):raise ValueError('Inference source differs from the trained checkpoint: '+name)
    if state.get('model_name')=='straight_residual_surface_process':
        from weld_pipeline.model_registry_residual import model_from_checkpoint as factory
    elif state.get('model_name')=='arc_adapter_surface_process':
        from weld_pipeline.model_registry_adapter import model_from_checkpoint as factory
    elif state.get('model_name')=='context_surface_process':
        from weld_pipeline.model_registry_context import model_from_checkpoint as factory
    elif state.get('model_name')=='wide_branch_surface':
        from weld_pipeline.model_registry_wide_branch import model_from_checkpoint as factory
    else:factory=model_from_checkpoint
    model=factory(state).cuda().eval()
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    torch.cuda.synchronize();start=time.perf_counter()
    xyz=read_xyz_pcd(pcd)*1000;read_seconds=time.perf_counter()-start
    sequence_function=make_robust_sequences if expected_version in ['typed_contact_surface_v3',CONTEXT_VERSION] else make_sequences
    if state.get('bounded_neighbors'):sequence_function=make_bounded_sequences
    if expected_version==MIXED_VERSION:
        if fast_mixed:
            from weld_pipeline.pooled_shared_mixed_sequences import make_fast_mixed_sequences
            sequences,geometry=make_fast_mixed_sequences(xyz)
        else:sequences,geometry=make_mixed_sequences(xyz)
    else:sequences,geometry=sequence_function(xyz)
    if surface and expected_version!=MIXED_VERSION:
        function=append_bidirectional_surface_features if expected_version==BOTH_FEATURE_VERSION else append_surface_features
        sequences,surface_stats=function(xyz,sequences)
        geometry['surface_features']=surface_stats
    if expected_version==CONTEXT_VERSION:
        if state.get('fast_context'):
            from weld_pipeline.frame_context_features_fast import append_frame_context_fast as context_function
        else:context_function=append_frame_context
        sequences,context_stats=context_function(xyz,sequences)
        geometry['frame_context']=context_stats
    extras=make_vertical_context(xyz,sequences)
    for sequence,extra in zip(sequences,extras):
        sequence['features']=ContextFeatures(sequence['features'],extra)
    geometry['wide_vertical_context']={'added_channels':81,'offsets_mm':[-150,-100,-50,-25,0,25,50,100,150],'source':'raw XYZ only'}
    feature_seconds=time.perf_counter()-start-read_seconds
    probabilities=infer_probabilities(model,sequences,halo=getattr(model,'inference_halo',256))
    if interval_parameters is not None:
        from weld_pipeline.event_gap_decoder import decode_intervals
        predictions=decode_intervals(sequences,probabilities,**interval_parameters)
    else:predictions=(decode_curved if expected_version==MIXED_VERSION else decode_typed)(sequences,probabilities)
    save_csv(predictions,out/'predicted_welds.csv');torch.cuda.synchronize();elapsed=time.perf_counter()-start
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
    if with_excel:save_all(predictions,out)
    runtime_sources=[Path(__file__),Path('weld_pipeline/export.py'),*[Path('weld_pipeline')/name for name in source_names],*extra_runtime_sources,*interval_sources]
    runtime_hashes={str(path):sha256(path) for path in runtime_sources}
    receipt={'input_pcd':str(pcd),'input_sha256':sha256(pcd),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
        'wide_feature_equivalence_receipt_sha256':sha256(wide_equivalence_path),
        'prediction_count':len(predictions),'raw_pcd_to_csv_seconds':elapsed,'pcd_read_seconds':read_seconds,
        'feature_seconds':feature_seconds,'model_load_timed':False,'excel_and_ply_timed':False,
        'repetitions':1,'reference_inputs_used':False,'feature_version':expected_version,'inference_halo':getattr(model,'inference_halo',256),
        'inference_source_hashes':runtime_hashes,
        'fast_mixed':fast_mixed,'pooled_query_workers':fast_mixed,'shared_curve_candidates':fast_mixed,'feature_equivalence_receipt_sha256':sha256(equivalence_path) if equivalence_path else None,'equivalence_scope':'Pooled base 84 channels; new 81 context channels checked separately.',
        'decoder_configuration':{'threshold':.5,'minimum_peak_separation_mm':4.,'minimum_weld_length_mm':3.,'internal_endpoint_margin_mm':1.5,
            'interval_parameters':interval_parameters,'configuration_sha256':sha256(decoder_config_path) if decoder_config_path else None},
        'scope':'Shared-model horizontal/vertical and circular-arc development prediction; performance requires independent scoring.' if expected_version==MIXED_VERSION else 'Shared-model straight horizontal/vertical development prediction; arc reconstruction unsupported.',
        'geometry':geometry}
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('RAW_PREDICTION',{k:receipt[k] for k in ['prediction_count','raw_pcd_to_csv_seconds','pcd_read_seconds','feature_seconds']},flush=True)
    return predictions,receipt


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pcd',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--outdir',required=True);ap.add_argument('--excel',action='store_true');ap.add_argument('--fast-mixed',action='store_true')
    ap.add_argument('--decoder-config');a=ap.parse_args()
    predict(a.pcd,a.checkpoint,a.outdir,a.excel,a.fast_mixed,a.decoder_config)
