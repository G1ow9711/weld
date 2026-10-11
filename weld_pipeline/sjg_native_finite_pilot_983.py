"""TRAIN21 learned-fragment geometry pilot; no target-assisted construction."""
import argparse
import builtins
import io
import inspect
from pathlib import Path
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_geometry_776 import serial
from weld_pipeline.sjg_xyz_batch_795 import contacts_from_support,save_surfaces
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_native_finite_geometry_982 import extract,controls


def diagnose(contacts,target):
    rows=[]
    for parent in target['parents']:
        for child in parent['children']:
            truth=CurveCarrier(child['representation']);samples=truth.at(truth.grid(1.));ends=truth.at(np.array([0.,truth.length]))
            options=[]
            for i,c in enumerate(contacts):
                pred=CurveCarrier(dict(kind='line',endpoints_mm=[c['p0'],c['p1']]))
                points=pred.at(pred.grid(1.));pends=pred.at(np.array([0.,pred.length]))
                directed=float(np.linalg.norm(pends-ends,axis=1).max())
                reverse=float(np.linalg.norm(pends[::-1]-ends,axis=1).max())
                d4=float(max(pred.project(samples)[1].max(),truth.project(points)[1].max()))
                options.append(dict(contact=i,length_mm=pred.length,endpoints_mm=pends.tolist(),
                    directed_endpoint_max_mm=directed,undirected_endpoint_max_mm=min(directed,reverse),
                    bidirectional_1mm_sample_max_mm=d4,face_indices=c['face_indices'],
                    geometry_only_pass=truth.kind=='line' and min(directed,reverse)<=3 and d4<=3))
            options.sort(key=lambda v:max(v['undirected_endpoint_max_mm'],v['bidirectional_1mm_sample_max_mm']))
            rows.append(dict(original_child_id=child['original_child_id'],kind=truth.kind,original_length_mm=truth.length,
                geometry_only_covered=any(v['geometry_only_pass'] for v in options),best_candidates=options[:5]))
    return dict(original_G=len(rows),geometry_only_covered=sum(r['geometry_only_covered'] for r in rows),rows=rows,
        geometric_carriers_not_weld_predictions=True,classification_and_parent_path_not_evaluated=True,
        reverse_allowed_only_for_representation_diagnosis=True,sample_step_mm=1.,full_contract_pass=False)


def main(extractor=extract,mesh_method='isotropic_knn',additional_controls=None):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    protocol=dict(stage='NATIVE_TRAIN21_LEARNED_FINITE_GEOMETRY_DIAGNOSTIC',source_key='21',
        backbone_run='run162',features='run166',instances='run173',source_sha256=digest(__file__),
        implementation_sha256=digest(__file__.replace('sjg_native_finite_pilot_983','sjg_native_finite_geometry_982')),
        extractor_sha256=digest(inspect.getfile(extractor)),mesh_method=mesh_method,
        plane_tolerance_mm=.2,mesh_grid_mm=1.,min_plane_points=16,normal_merge_angle_deg=1.,
        full_original_support_saved_separately=True,no_hardcoded_weld_length_or_endpoint_trim=True,
        no_STEP_or_target_reads_during_construction=True,other_eight_geometries_NOT_RUN=True,
        original_eight_G_unchanged=496,supplemental_BM_G=4,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        write(dest/'controls.json',controls());check_resources()
        if additional_controls is not None:write(dest/'additional_controls.json',additional_controls())
        case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
        feature=read(OUT/'run166/source_21_freeze.json');partition=read(OUT/'run173/receipt.json')['sources'][0]
        assert case['input_sha256']==feature['input_sha256']==partition['input_sha256']
        paths=[OUT/case['input_file'],OUT/'run166'/feature['feature_file'],OUT/'run173'/partition['file']]
        hashes=[case['input_sha256'],feature['feature_sha256'],partition['sha256']]
        for path,sha in zip(paths,hashes):assert digest(path)==sha
        original_open,original_io=builtins.open,io.open
        def guard(factory):
            def opened(path,*a,**kw):
                if isinstance(path,(str,bytes,Path)):
                    value=str(path).lower()
                    if value.endswith(('.step','.stp','.xlsx')) or any(s in value for s in ('teacher','target','supervision','geometry.json')):
                        raise RuntimeError('Reference read forbidden during finite geometry: '+str(path))
                return factory(path,*a,**kw)
            return opened
        started=time.perf_counter()
        try:
            builtins.open=guard(original_open);io.open=guard(original_io)
            xyz=np.load(paths[0],mmap_mode='r')
            with np.load(paths[1]) as z:pred={k:z[k] for k in ('normal','boundary')}
            with np.load(paths[2]) as z:instance=z['face_instance'];material=z['material_instance']
            def progress(*v):
                check_resources();print('NATIVE_FINITE_21',*v,flush=True)
                write(dest/'progress.json',dict(event=list(v),time_utc=utc()))
            faces,full,unresolved,stats=extractor(xyz,pred,instance,material,progress)
            np.savez(dest/'original_surface_support.npz',**full);np.save(dest/'unresolved_original_point_ids.npy',unresolved)
            save_surfaces(dest,faces);contacts=contacts_from_support(faces,xyz)
            write(dest/'contacts.json',serial(contacts))
            files={name:digest(dest/name) for name in ('surfaces.json','surface_support.npz','original_surface_support.npz','unresolved_original_point_ids.npy','contacts.json')}
            write(dest/'input_freeze.json',dict(**stats,source_key='21',input_sha256=case['input_sha256'],
                feature_sha256=feature['feature_sha256'],instance_sha256=partition['sha256'],files=files,
                contacts=len(contacts),python_reference_reads_during_construction=0,native_file_API_not_instrumented=True,
                geometry_seconds=time.perf_counter()-started,full_pipeline_uncached_timing_not_measured=True))
        finally:builtins.open=original_open;io.open=original_io
        target=OUT/case['target_file'];assert digest(target)==case['target_sha256']
        result=diagnose(contacts,read(target));write(dest/'geometry_diagnosis.json',result)
        write(dest/'receipt.json',dict(status='NATIVE_TRAIN21_FINITE_GEOMETRY_DIAGNOSED',stats=stats,
            contacts=len(contacts),geometry_only_covered=result['geometry_only_covered'],original_G=result['original_G'],
            weld_extraction_accuracy_not_measured=True,full_contract_pass=False,end_utc=utc()))
        print('NATIVE_FINITE_GEOMETRY_DIAGNOSED','CARRIERS',len(contacts),'GEOMETRY_ONLY',result['geometry_only_covered'],'/',result['original_G'],flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
