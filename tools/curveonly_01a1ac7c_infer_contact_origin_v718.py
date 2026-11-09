"""Curve inference with frozen weights and a verified numerical contact fix."""
from __future__ import annotations
import argparse,importlib,json,os,pickle,time
from pathlib import Path
from tools import curveonly_01a1ac7c_infer_frozen_v536 as package
REC=package.ROOT/'experiment_records/CURVEONLY_01a1ac7c_20261106_v1/contact_origin_control_v712_20261109'
DEFAULT=REC/'INFERENCE_SPEC.json'
RUNTIME='weld_pipeline.curveonly_01a1ac7c_contact_origin_runtime_v711'


def freeze():
    from tools import curveonly_01a1ac7c_contact_origin_control_v712 as control
    p=control.check();read=lambda f:json.loads((REC/f).read_text())
    assert read('SCORE_FULL_ACTUAL.json')['exit_code']==0 and read('SCORES_FULL.json')['passed']
    assert read('GEOMETRY_CONTROLS717_ACTUAL.json')['exit_code']==0 and read('GEOMETRY_CONTROLS.json')['all_passed']
    assert read('RAW_ACTUAL.json')['exit_code']==0
    raw=json.loads((control.OUT/'raw_2D2/RECEIPT.json').read_text());assert raw['control_parents_exact']and raw['original_three_parents_exact']
    original=package.verify(package.DEFAULT);sources={b['path']:b for b in original['runtime_sources']}
    for path in [package.ROOT/'weld_pipeline/curveonly_01a1ac7c_contact_origin_v710.py',package.ROOT/'weld_pipeline/curveonly_01a1ac7c_contact_origin_runtime_v711.py',Path(__file__).resolve()]:
        b=dict(path=str(path.relative_to(package.ROOT)),sha256=package.sha(path),bytes=path.stat().st_size);sources[b['path']]=b
    spec=dict(schema='CURVEONLY_VERIFIED_CONTACT_ORIGIN_V718',runtime=RUNTIME,weights=original['weights'],sources=list(sources.values()),
        original_spec=dict(path=str(package.DEFAULT.relative_to(package.ROOT)),sha256=package.sha(package.DEFAULT),bytes=package.DEFAULT.stat().st_size),
        input_contract=original['input_contract'],change='Only recover missed finite shell hits within the pre-existing 1e-6mm ray-origin band, under unchanged original complete T-joint and final-face checks.',
        old_TRAIN=dict(inputs=22,TP=121,G=121,FP=0,FN=0),old_VAL=dict(inputs=9,TP=37,G=37,FP=0,FN=0),
        uploaded_three=dict(TP=10,G=10,FP=0,FN=0),uploaded_2D2=dict(TP=4,G=4,FP=0,FN=0),
        uploaded_two_repeat_historical_VAL=True,no_2D2_supervised_fitting=True,original_weights_and_237_sources_unchanged=True,
        complete_curve_geometry_ignore_internal_representation_cuts=True,endpoint_and_whole_tolerance_mm=3.,
        actual_raw_2D2_control_parity_passed=True,geometry_rule_controls_passed=13,
        entire_model_coordinate_invariance_and_unseen_should_weld_generalization_not_certified=True,
        no_contract_independent_or_cold_performance_acceptance_claim=True)
    assert not DEFAULT.exists();package.save(DEFAULT,spec);print('CONTACT_ORIGIN_SPEC_FROZEN',len(sources),flush=True)


def verify(path):
    s=json.loads(Path(path).read_text());assert s['schema']=='CURVEONLY_VERIFIED_CONTACT_ORIGIN_V718'and s['runtime']==RUNTIME
    package.bound(s['weights']);package.bound(s['original_spec'])
    for b in s['sources']:package.bound(b)
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--verify-only',action='store_true')
    p.add_argument('--spec',type=Path,default=DEFAULT);p.add_argument('--xyz',type=Path);p.add_argument('--step',type=Path);p.add_argument('--pose-config',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--workpiece-id',default='workpiece');a=p.parse_args()
    if a.freeze:freeze();return
    s=verify(a.spec)
    if a.verify_only:print('CONTACT_ORIGIN_SPEC_VERIFIED',s['weights']['sha256'],len(s['sources']));return
    if not(a.xyz and a.out):p.error('--xyz and --out are required')
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    pose=None
    if a.pose_config:
        config=json.loads(a.pose_config.read_text());assert not(set(config)-{'units','coordinate_frame','multiplication','transform_STEP_to_XYZ'})and config.get('units')=='mm'
        pose=config['transform_STEP_to_XYZ'];assert len(pose)==4 and all(len(row)==4 for row in pose)
    raw=dict(path=str(a.xyz.resolve()),sha256=package.sha(a.xyz));step=dict(path=str(a.step.resolve()),sha256=package.sha(a.step))if a.step else None
    from tools.curveonly_01a1ac7c_paired_face_control_v409 import install_fence
    from threadpoolctl import threadpool_limits
    install_fence();runtime=importlib.import_module(RUNTIME)
    with package.bound(s['weights']).open('rb')as f:model=pickle.load(f)
    a.out.mkdir(parents=True,exist_ok=True);assert not(a.out/'curves.json').exists();started=time.perf_counter()
    with threadpool_limits(limits=2):result=runtime.extract(dict(raw=raw,STEP=step,transform=pose),model)
    package.export_complete(a.out,a.workpiece_id,result)
    package.save(a.out/'run.json',dict(status='SUCCESS',runtime=RUNTIME,model_spec_sha256=package.sha(a.spec),raw=raw,STEP=step,
        weights=s['weights'],parents=len(result['parents']),seconds=time.perf_counter()-started,
        contact_origin_rule=result.get('contact_origin_rule'),segmentation_not_evaluated=True,no_independent_acceptance_claim=True))
    print('CONTACT_ORIGIN_INFERENCE_CLOSED',len(result['parents']),str(a.out),flush=True)


if __name__=='__main__':main()
