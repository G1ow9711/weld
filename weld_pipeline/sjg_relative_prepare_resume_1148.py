"""Resume only unfinished pure inputs after preserving run341's resource failure."""
import argparse,gc,shutil,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS,feature_path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args();dest=OUT/'run344'
    if args.worker:
        from weld_pipeline.sjg_relative_geometry_batch_1136 import prepare as relative
        from weld_pipeline.sjg_observed_face_features_1138 import prepare as observed
        started=time.perf_counter();base=dest/'remaining142';base.mkdir()
        try:
            from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
            profilefile=OUT/'run185/source_21/profiles.json'
            original=read(profilefile);compact=read_profile_geometry(profilefile)
            assert compact==[dict(representation=p['representation'],face_indices=p.get('face_indices',[])) for p in original]
            del original,compact;gc.collect()
            # Exact control of changed IO lifetime on one complete source, no labels.
            control=dest/'io_control';control.mkdir()
            observed('21',control,prior_run='run327')
            a=np.load(control/'source_21/features.npy',mmap_mode='r');b=np.load(OUT/'run329/source_21/features.npy',mmap_mode='r')
            assert np.array_equal(a,b);a._mmap.close();b._mmap.close()
            for key in ('25','36','42'):
                check_resources()
                if key=='25':prior='run340'
                else:
                    relative(key,base);prior=str(base.relative_to(OUT));gc.collect()
                observed(key,dest,prior_run=prior);gc.collect()
            cases=[]
            for key in KEYS:
                folder=feature_path(key);r=read(folder/'receipt.json')
                assert r['columns']==152 and digest(folder/'features.npy')==r['features_sha256']
                cases.append(dict(source=key,feature_run=folder.parent.name,receipt_sha256=digest(folder/'receipt.json'),**{k:v for k,v in r.items() if k!='source'}))
            write(dest/'all_relative_inputs_frozen_before_labels.json',dict(cases=cases,GT_reads=0,STEP_reads=0))
            write(dest/'worker_receipt.json',dict(status='SEVEN_RELATIVE_INPUTS_FROZEN',cases=cases,fit_calls=0,GT_reads=0,STEP_reads=0,
                seconds=time.perf_counter()-started,feature_definition_changed=False,io_lifetime_exact_control=True,prior_failures_preserved=['run341','run343']))
        except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise
        return
    assert read(OUT/'run343/failure.json')['watchdog']['exit_code']!=0
    check_resources();dest.mkdir(exist_ok=False);snap=dest/'source_snapshot';snap.mkdir()
    for name in ('sjg_relative_prepare_resume_1148.py','sjg_relative_geometry_batch_1136.py','sjg_observed_face_features_1138.py','sjg_stream_profile_geometry_1149.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snap)
    write(dest/'registered_protocol.json',dict(status='REMAINING_INPUTS_LOWER_MMAP_RESIDENCY',sources=['25','36','42'],
        reused_complete_sources=['21','4','10','BM79A_native_local'],reused_142_source='25',columns=152,fit_calls=0,wall_seconds=180,
        original_resource_guards=True,previous_failure_not_erased=True,utc=utc()))
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_prepare_resume_1148','--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='REMAINING_RELATIVE_INPUTS_INCOMPLETE',watchdog=result));raise RuntimeError('Input preparation incomplete')
    r=read(dest/'worker_receipt.json');r['external_wall_seconds']=result['wall_seconds'];write(dest/'receipt.json',r)
    print('SEVEN_RELATIVE_INPUTS_READY',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
