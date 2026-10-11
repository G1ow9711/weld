"""Reference CAD contact/normal audit of GR/FR labels and point candidates."""
import json
from pathlib import Path

import gmsh
import numpy as np

from weld_pipeline.curved_cache import load_piece
from weld_pipeline.truth import load_truth


def outward(volume, face):
    tag = face['face']
    lower, upper = gmsh.model.getParametrizationBounds(2, tag)
    for f in [.5, .3, .7, .15, .85]:
        uv = np.array(lower) + f * (np.array(upper) - lower)
        if not gmsh.model.isInside(2, tag, uv.tolist(), parametric=True):
            continue
        p = np.array(gmsh.model.getValue(2, tag, uv.tolist()))
        n = np.array(face['parametric_normal'])
        plus = gmsh.model.isInside(3, volume, (p + .01 * n).tolist())
        minus = gmsh.model.isInside(3, volume, (p - .01 * n).tolist())
        if plus != minus:
            return {'normal': (n if minus else -n).tolist(), 'verified': True,
                    'method': 'Solid membership at face interior +/-0.01 mm'}
    return {'normal': None, 'verified': False, 'method': 'No verified face-interior normal sample'}


def contains(face, point):
    q, uv = gmsh.model.getClosestPoint(2, face['face'], point.tolist())
    if np.linalg.norm(np.array(q) - point) > .2:
        return False
    if gmsh.model.isInside(2, face['face'], uv.tolist(), parametric=True):
        return True
    # Registration/point export rounding can put a true boundary a few microns
    # outside the trim. Admit only a tiny move towards the planar face center.
    center = np.array(face['center_mm'])
    delta = center - np.array(q)
    delta -= np.array(face['parametric_normal']) * np.dot(delta, face['parametric_normal'])
    if np.linalg.norm(delta) < 1e-8:
        return False
    test = np.array(q) + delta / np.linalg.norm(delta) * .1
    qq, uv = gmsh.model.getClosestPoint(2, face['face'], test.tolist())
    return bool(gmsh.model.isInside(2, face['face'], uv.tolist(), parametric=True))


def contacts(point, faces, rotation, translation):
    local = (point - translation) @ rotation
    found = []
    for face in faces:
        n = np.array(face['parametric_normal'])
        distance = abs(np.dot(local - face['center_mm'], n))
        if distance > .2 or not contains(face, local):
            continue
        signed = face['outward']
        found.append({'volume': face['volume'], 'board_name': face['board_name'], 'face': face['face'],
                      'plane_distance_mm': float(distance), 'thickness_candidate_mm': face['thickness'],
                      'unsigned_normal_world': (rotation @ n).tolist(),
                      'outward_normal_world': (rotation @ np.array(signed['normal'])).tolist() if signed['verified'] else None,
                      'outward_verified': signed['verified']})
    return found


def main():
    root = Path('weld_pipeline/reports/step_alignment_20260930')
    split = json.loads(Path('weld_pipeline/data/contract_child_v1/split.json').read_text('utf8'))
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
    try:
        for name in ['224-GR24A1', '527-FR146A']:
            folder = Path('DataSet/9月20日') / name
            cad = json.loads((Path('weld_pipeline/reports/step_geometry_20260930') / (name + '.json')).read_text('utf8'))
            pose_file = root / (name + '_refined.json')
            if pose_file.exists():
                pose = json.loads(pose_file.read_text('utf8'))['trials'][0]['transform']
            else:
                pose_file = root / (name + '.json')
                pose = json.loads(pose_file.read_text('utf8'))['alignment_trials'][0]['transform']
            transform = np.array(pose)
            rotation, translation = transform[:3, :3], transform[:3, 3]
            gmsh.clear()
            gmsh.model.occ.importShapes(str(next(folder.glob('*.stp')).resolve()))
            gmsh.model.occ.synchronize()
            faces = []
            for volume in cad['volumes']:
                for face in volume['planar_faces']:
                    faces.append({**face, 'volume': volume['volume'], 'board_name': volume['name'],
                                  'thickness': volume['supported_thickness_mm'], 'outward': outward(volume['volume'], face)})
            truth, excluded = load_truth(folder)
            welds = []
            for w in truth:
                if w['type'] != 'vertical':
                    continue
                locations = []
                for si, segment in enumerate(w['segments']):
                    p = np.array(segment['points'])
                    for label, q in [('mid', (p[0] + p[-1]) / 2.), ('end', p[-1])]:
                        locations.append({'segment': si, 'location': label, 'xyz_mm': q.tolist(),
                                          'contacts': contacts(q, faces, rotation, translation)})
                welds.append({'id': w['id'], 'number': w['metadata']['WeldNumber (f/v + Name number)'],
                              'segments': len(w['segments']), 'side_plate_split_any': w['metadata']['IsSplitedForDualRobot (Any)'],
                              'locations': locations})
            piece = load_piece(split['piece_roots'][name], verify=True)
            mapping = piece['record']['mapping']['line_mapping']['weld_to_sequence']
            positive = {int(mapping[str(i)]) for i, w in enumerate(piece['truth']) if w['type'] == 'vertical'}
            candidates = []
            for i, seq in enumerate(piece['sequences']):
                if seq['type'] != 'vertical':
                    continue
                q = (seq['xyz'][0].astype(float) + seq['xyz'][-1].astype(float)) / 2.
                found = contacts(q, faces, rotation, translation)
                angles = []
                for f in found:
                    normal = np.array(f['unsigned_normal_world'])
                    old_axes = seq['features'][len(seq['features']) // 2, 60:66].reshape(2, 3)
                    angles.append(float(np.degrees(np.arccos(np.clip(np.abs(old_axes @ normal).max(), -1., 1.)))))
                candidates.append({'sequence': i, 'reference_positive_diagnostic_only': i in positive,
                                   'midpoint_xyz_mm': q.tolist(), 'contact_board_count': len({f['volume'] for f in found}),
                                   'contacts': found, 'angles_to_old_global_axes_degrees': angles})
            report = {'name': name, 'pose_file': str(pose_file), 'scope': 'Reference-assisted structural diagnosis, not inference rules or reviewed weld-side attribute truth.',
                      'normal_method': 'Native CAD surface direction corrected using solid inside/outside membership; boundary signs alone failed the translated-box check.',
                      'welds': welds, 'candidates': candidates,
                      'faces_with_verified_outward': sum(f['outward']['verified'] for f in faces), 'planar_faces': len(faces)}
            (root / (name + '_contacts.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            print('CONTACT_AUDIT', name, 'outward', report['faces_with_verified_outward'], '/', len(faces),
                  'welds', len(welds), 'candidate_board_counts',
                  [(c['sequence'], c['reference_positive_diagnostic_only'], c['contact_board_count'],
                    [round(a, 3) for a in c['angles_to_old_global_axes_degrees']]) for c in candidates], flush=True)
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    main()
