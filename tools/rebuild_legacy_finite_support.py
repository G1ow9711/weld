"""Rebuild the learned finite-support branch of 999; legacy XYZ union is separate."""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install

ADAPTER = install()
from weld_pipeline.sjg_local_runtime_904 import digest, write, check_resources
from weld_pipeline.sjg_ideal_plane_support_997 import extract, controls
from weld_pipeline.sjg_xyz_batch_795 import contacts_from_support, save_surfaces
from weld_pipeline.sjg_native_joint_candidates_999 import contact_profiles
from weld_pipeline.sjg_unified_local_candidates_950 import save_graph_faces
from weld_pipeline.sjg_geometry_776 import serial

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/best_baseline_restore'


def prepare(key):
    ins = BASE / 'instances' / ('source_' + key)
    features = BASE / 'merged_point_features' / ('source_' + key)
    ir = json.loads((ins / 'receipt.json').read_text())
    fr = json.loads((features / 'receipt.json').read_text())
    assert ir['status'] == 'ORIGINAL_INSTANCE_PIPELINE_RECONSTRUCTED'
    assert ir['input_sha256'] == fr['input_sha256']
    xyzfile = ROOT / fr['input_file']
    assert digest(xyzfile) == fr['input_sha256']
    assert digest(features / fr['feature_file']) == fr['feature_sha256']
    assert digest(ins / ir['file']) == ir['sha256']
    dest = BASE / 'learned_finite_support' / ('source_' + key)
    dest.mkdir(parents=True, exist_ok=False)
    mesh = dest / 'finite_mesh'; mesh.mkdir()
    write(dest / 'controls.json', controls())
    write(dest / 'protocol.json', dict(source_sha256=digest(__file__), adapter=ADAPTER,
          input_sha256=fr['input_sha256'], point_features_sha256=fr['feature_sha256'], instances_sha256=ir['sha256'],
          original_modules={n:digest(ROOT/'weld_pipeline'/n) for n in (
              'sjg_ideal_plane_support_997.py', 'sjg_native_finite_geometry_982.py',
              'sjg_union_sampling_mesh_995.py', 'sjg_native_joint_candidates_999.py')},
          original_run132_xyz_union_pending=True, label_and_STEP_reads=0))
    started = time.perf_counter()
    try:
        xyz = np.load(xyzfile, mmap_mode='r')
        with np.load(features / fr['feature_file']) as z:
            pred = {k:z[k] for k in ('normal', 'boundary')}
        with np.load(ins / ir['file']) as z:
            instance, material = z['face_instance'], z['material_instance']
        def progress(*event):
            check_resources()
            write(dest / 'progress.json', dict(event=list(event), seconds=time.perf_counter()-started))
            print('ORIGINAL_FINITE_SUPPORT', key, *event, flush=True)
        faces, full, left, stats = extract(xyz, pred, instance, material, progress)
        np.savez_compressed(mesh / 'original_surface_support.npz', **full)
        np.save(mesh / 'unresolved_original_point_ids.npy', left)
        save_surfaces(mesh, faces)
        contacts = contacts_from_support(faces, xyz)
        write(dest / 'contacts.json', serial(contacts))
        profiles = contact_profiles(contacts, faces)
        for face in faces:
            face['point_ids'] = full[face['full_support_key']]
            face['graph_membership_uses_full_original_support'] = True
        save_graph_faces(dest, faces)
        write(dest / 'profiles.json', serial(profiles))
        write(dest / 'receipt.json', dict(status='LEARNED_FINITE_BRANCH_RECONSTRUCTED', source_key=key,
              input_sha256=fr['input_sha256'], statistics=stats, contacts=len(contacts), profiles=len(profiles),
              files={n:digest(dest/n) for n in ('surfaces.json', 'surface_support.npz', 'profiles.json', 'contacts.json')},
              seconds=time.perf_counter()-started, original_run132_xyz_union_pending=True,
              historical_parity_proven=False, weld_accuracy_measured=False, GT_reads=0, STEP_reads=0))
        print('ORIGINAL_FINITE_COMPLETE', key, stats, 'profiles', len(profiles), flush=True)
    except BaseException:
        write(dest / 'failure.json', dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, choices=['21', 'BM79A_native_local', '10', '25', '36'])
    args = ap.parse_args()
    with threadpool_limits(limits=2):
        prepare(args.key)
