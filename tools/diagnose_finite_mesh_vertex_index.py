"""Capture the actual input-only finite mesh that raises a vertex IndexError.

The existing extractor runs unchanged in an isolated function namespace.
Predicted point features and memberships are authenticated before use. Raw
geometry stays in a private NPZ artifact; labels and STEP geometry are unused.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import FunctionType
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline import sjg_ideal_plane_support_997 as ideal
from weld_pipeline import sjg_union_sampling_mesh_995 as mesh
from weld_pipeline.sjg_geometry_776 import serial
from weld_pipeline.sjg_local_runtime_904 import read, write, digest, check_resources
from tools.prepare_full_training_case import registered_job, roots, ROOT, BASE


class CapturedMeshFailure(Exception):
    pass


def main(args):
    if Path(args.experiment_name).name!=args.experiment_name:
        raise ValueError('An independent local diagnostic name is required')
    out=BASE/args.experiment_name;out.mkdir(exist_ok=False)
    job=registered_job(args.job)
    work,frozen,inputs=roots(job);key=job['runtime_key']
    features=frozen/'merged_point_features'/('source_'+key)
    instances=frozen/'instances'/('source_'+key)
    fr,ir=read(features/'receipt.json'),read(instances/'receipt.json')
    xyzfile=ROOT/fr['input_file']
    if (fr['input_sha256']!=ir['input_sha256'] or digest(xyzfile)!=fr['input_sha256']
            or digest(features/fr['feature_file'])!=fr['feature_sha256']
            or digest(instances/ir['file'])!=ir['sha256']):
        raise ValueError('Original predicted input data changed')
    write(out/'protocol.json',dict(job_key=args.job,runtime_key=key,
        registry_sha256=digest(BASE/'training_registry.json'),xyz_sha256=fr['input_sha256'],
        point_features_sha256=fr['feature_sha256'],instances_sha256=ir['sha256'],
        scipy_version=scipy.__version__,source_sha256=digest(__file__),
        original_mesh_sha256=digest(ROOT/'weld_pipeline/sjg_union_sampling_mesh_995.py'),
        original_extractor_sha256=digest(ROOT/'weld_pipeline/sjg_native_finite_geometry_982.py'),
        original_ideal_support_sha256=digest(ROOT/'weld_pipeline/sjg_ideal_plane_support_997.py'),
        label_geometry_reads=0,STEP_reads=0,optimizer_updates=0))
    xyz=np.load(xyzfile,mmap_mode='r')
    with np.load(features/fr['feature_file']) as z:
        pred={k:z[k] for k in ('normal','boundary')}
    with np.load(instances/ir['file']) as z:
        instance,material=z['face_instance'],z['material_instance']
    prefix=[];calls=0;begin=time.perf_counter()
    def capture(points,plane):
        nonlocal calls
        calls+=1
        try:
            face=mesh.local_support(points,plane)
        except IndexError as exc:
            ids,uv=mesh.projected_unique(points,plane)
            radius,used=mesh.two_dimensional_radius(uv)
            triangulation=mesh.Delaunay(uv)
            triangles=triangulation.simplices
            np.savez_compressed(out/'failed_mesh.npz',xyz=points[plane['point_ids']],
                normal=plane['normal'],center=plane['center'],offset=np.asarray(plane['offset']),
                basis=plane['basis'],projected_uv=uv,projected_point_ids=ids,
                radius=radius,used=used,triangles=triangles)
            write(out/'valid_prefix_meshes.json',prefix)
            write(out/'receipt.json',dict(status='ACTUAL_ORIGINAL_INPUT_MESH_VERTEX_INDEX_ERROR_CAPTURED',
                job_key=args.job,mesh_call=calls,prior_mesh_calls=len(prefix),error=str(exc),
                projected_vertices=len(uv),triangles_shape=list(triangles.shape),
                minimum_vertex_index=int(triangles.min()),maximum_vertex_index=int(triangles.max()),
                invalid_triangle_count=int(((triangles<0)|(triangles>=len(uv))).any(1).sum()),
                rank_supported_vertices=int((used>0).sum()),rank_supported_cells=bool((used>0).any()),
                private_geometry_sha256=digest(out/'failed_mesh.npz'),
                prefix_mesh_fingerprints_sha256=digest(out/'valid_prefix_meshes.json'),
                protocol_sha256=digest(out/'protocol.json'),seconds=time.perf_counter()-begin,
                labels_or_model_parameters_changed=False,complete_input_reconstruction=False))
            raise CapturedMeshFailure from exc
        encoded=json.dumps(serial(face),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        prefix.append(dict(mesh_call=calls,face_present=face is not None,
                           original_mesh_sha256=hashlib.sha256(encoded).hexdigest()))
        return face
    environment=dict(ideal.extract.__globals__);environment['local_support']=capture
    extract=FunctionType(ideal.extract.__code__,environment,ideal.extract.__name__,
                         ideal.extract.__defaults__,ideal.extract.__closure__)
    def progress(*event):
        check_resources()
        write(out/'progress.json',dict(event=list(event),seconds=time.perf_counter()-begin))
        if event[0]=='FINITE_MESH' and event[1]%100==0:print('MESH_DIAGNOSTIC_PROGRESS',*event,flush=True)
    try:
        with threadpool_limits(limits=2):
            extract(xyz,pred,instance,material,progress)
    except CapturedMeshFailure:
        print('ACTUAL_MESH_VERTEX_FAILURE_CAPTURED',calls,flush=True)
        return
    raise RuntimeError('The original failure was not reproduced; do not claim a captured control')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--experiment-name',required=True)
    main(parser.parse_args())
