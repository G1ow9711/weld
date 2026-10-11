"""Verify the real unsupported mesh and every preceding original mesh output."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
from types import FunctionType, SimpleNamespace
import time

import numpy as np

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline import sjg_union_sampling_mesh_995 as mesh
from weld_pipeline.sjg_geometry_776 import serial
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from tools import diagnose_finite_mesh_vertex_index as diagnostic

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/full_dataset_v1'


class VerifiedPrefix(Exception):
    pass


def main(args):
    if Path(args.experiment_name).name!=args.experiment_name:
        raise ValueError('An independent local control name is required')
    captured=Path(args.diagnostic).resolve();captured.relative_to(ROOT)
    output=BASE/args.experiment_name;output.mkdir(exist_ok=False)
    receipt,protocol=read(captured/'receipt.json'),read(captured/'protocol.json')
    if (receipt['status']!='ACTUAL_ORIGINAL_INPUT_MESH_VERTEX_INDEX_ERROR_CAPTURED'
            or digest(captured/'protocol.json')!=receipt['protocol_sha256']
            or digest(captured/'failed_mesh.npz')!=receipt['private_geometry_sha256']
            or digest(captured/'valid_prefix_meshes.json')!=receipt['prefix_mesh_fingerprints_sha256']
            or digest(captured/'original_mesh_source.txt')!=protocol['original_mesh_sha256']):
        raise ValueError('The original actual failure control changed')
    with np.load(captured/'failed_mesh.npz',allow_pickle=False) as z:
        xyz=z['xyz'];plane=dict(point_ids=np.arange(len(xyz)),normal=z['normal'],
            center=z['center'],basis=z['basis'],offset=float(z['offset']))
        assert np.count_nonzero(z['used']>0)==0
    frozen=runpy.run_path(str(captured/'original_mesh_source.txt'))
    try:
        frozen['local_support'](xyz,plane)
    except IndexError as exc:
        assert str(exc)==receipt['error']
    else:
        raise AssertionError('The frozen original actual failure did not reproduce')
    before=digest(captured/'failed_mesh.npz')
    assert mesh.local_support(xyz,plane) is None
    assert digest(captured/'failed_mesh.npz')==before
    normal_controls=mesh.controls()
    prefix=read(captured/'valid_prefix_meshes.json');checked=0;begin=time.perf_counter()
    helper_sha=digest(ROOT/'weld_pipeline/sjg_union_sampling_mesh_995.py')
    def checked_mesh(points,actual_plane):
        nonlocal checked
        face=mesh.local_support(points,actual_plane);checked+=1
        if checked<=len(prefix):
            encoded=json.dumps(serial(face),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            assert hashlib.sha256(encoded).hexdigest()==prefix[checked-1]['original_mesh_sha256'],checked
        elif checked==receipt['mesh_call']:
            assert face is None,'Unsupported actual points were turned into a surface'
            raise VerifiedPrefix
        else:
            raise AssertionError('The actual failed-mesh position changed')
        return face
    environment=dict(diagnostic.main.__globals__)
    environment['mesh']=SimpleNamespace(local_support=checked_mesh)
    replay=FunctionType(diagnostic.main.__code__,environment,diagnostic.main.__name__,
                        diagnostic.main.__defaults__,diagnostic.main.__closure__)
    try:
        replay(SimpleNamespace(job=receipt['job_key'],experiment_name=args.experiment_name+'_prefix_replay'))
    except VerifiedPrefix:
        pass
    else:
        raise AssertionError('Original valid prefix was not completely verified')
    assert checked==receipt['mesh_call'] and len(prefix)==checked-1
    assert helper_sha==digest(ROOT/'weld_pipeline/sjg_union_sampling_mesh_995.py')
    write(output/'receipt.json',dict(status='PASS_ACTUAL_UNSUPPORTED_MESH_AND_ALL_ORIGINAL_PREFIX_RESULTS',
        job_key=receipt['job_key'],original_failure_reproduced=True,unsupported_rank_vertices=0,
        unsupported_face_remains_unresolved=True,valid_prefix_mesh_calls=len(prefix),
        all_prior_meshes_bitwise_equal=True,original_density_boundary_controls=normal_controls,
        raw_point_coordinates_or_labels_changed=False,whole_workpiece_skipped=False,
        original_mesh_source_sha256=protocol['original_mesh_sha256'],new_mesh_source_sha256=helper_sha,
        diagnostic_receipt_sha256=digest(captured/'receipt.json'),source_sha256=digest(__file__),
        elapsed_prefix_seconds=time.perf_counter()-begin,GT_geometry_reads=0,optimizer_updates=0,
        full_dataset_or_model_quality_not_claimed=True))
    print('ACTUAL_FINITE_RANK_GUARD_AND_PREFIX_PARITY_PASS',len(prefix),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic',required=True)
    parser.add_argument('--experiment-name',required=True)
    main(parser.parse_args())
