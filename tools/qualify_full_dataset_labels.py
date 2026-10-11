"""Qualify every assigned dataset version without changing its split.

PLY and Excel are primary evidence. Modern ordered geometry is cross-checked
against both; native CBF boundaries are auxiliary and must also pass those
checks. Only a unique geometry bijection can repair changed export ObjectIds.
Unknown process attributes and interior endpoint frames remain masked.
This is offline label preparation, never inference or test-based model tuning.
"""
import argparse
import collections
import copy
import gc
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import openpyxl
from scipy.spatial import cKDTree

from tools.rebuild_expansion_excel_endpoint_sidecars_v2 import FIELDS, numeric, endpoint
from tools.compile_expansion_endpoint_attribute_labels import FIELDS as ATTRIBUTE_FIELDS, ENCODING
from weld_pipeline.cbf_weld_label_reader_332_v2 import extract_cbf_weld_json, JsonReferenceIndex
from weld_pipeline.contract_curve_geometry import fit_ordered_circle
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'
DATA = ROOT.parent / 'data/G_Data'
ENDPOINT_TOLERANCE_MM = 1e-4
LENGTH_TOLERANCE_MM = 1e-3


def workbook_rows(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        rows = book['焊缝信息'].iter_rows(values_only=True)
        headers = [str(value or '').split('\n')[-1] for value in next(rows)]
        if len(set(headers)) != len(headers):
            raise ValueError('Duplicate normalized workbook columns')
        values = [r for v in rows if (r := dict(zip(headers, v))).get('ObjectId')]
    finally:
        book.close()
    if len({str(r['ObjectId']) for r in values}) != len(values):
        raise ValueError('Duplicate Excel parent identifiers')
    return {str(r['ObjectId']): r for r in values}


def ply_header(path):
    header = []
    with path.open() as stream:
        for line in stream:
            header.append(line.strip())
            if line.strip() == 'end_header':
                break
    if 'format ascii 1.0' not in header or 'comment unit millimeter; frame reference_workpiece' not in header:
        raise ValueError('Unconfirmed PLY format, units or coordinate frame')
    count = int(next(s.split()[-1] for s in header if s.startswith('element vertex')))
    if count <= 0:
        raise ValueError('Empty workpiece or weld PLY')
    return header, count


def ply_points(path):
    header, count = ply_header(path)
    with path.open() as stream:
        for line in stream:
            if line.strip() == 'end_header':
                break
        points = np.loadtxt(stream, usecols=(0, 1, 2), ndmin=2, dtype=np.float64)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError('PLY count or finite XYZ validation failed')
    return points


def row_endpoint(row, side):
    value = np.asarray([row[side + 'Matrix.Translation.' + a] for a in 'XYZ'], float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Excel parent endpoint missing or nonfinite')
    return value


def bind_geometry(welds, rows):
    if len(welds) != len(rows) or len({str(w['id']) for w in welds}) != len(welds):
        raise ValueError(f'Geometry/Excel parent cardinality conflict: {len(welds)} vs {len(rows)}')
    same_ids = {str(w['id']) for w in welds} == set(rows)
    binding = {}
    for weld in welds:
        segments = weld['segments']
        if not segments:
            raise ValueError('Empty parent geometry')
        first, last = np.asarray(segments[0]['points'][0]), np.asarray(segments[-1]['points'][-1])
        length = sum(float(s['lengthMm']) for s in segments)
        candidates = [str(weld['id'])] if same_ids else list(rows)
        matches = [pid for pid in candidates if int(rows[pid]['SegmentCount']) == len(segments)
                   and abs(float(rows[pid]['Length']) - length) < LENGTH_TOLERANCE_MM
                   and np.linalg.norm(row_endpoint(rows[pid], 'Start') - first) < ENDPOINT_TOLERANCE_MM
                   and np.linalg.norm(row_endpoint(rows[pid], 'End') - last) < ENDPOINT_TOLERANCE_MM]
        if len(matches) != 1:
            raise ValueError(f'Parent {weld["id"]} has {len(matches)} exact geometry/Excel correspondences')
        binding[str(weld['id'])] = matches[0]
    if set(binding.values()) != set(rows):
        raise ValueError('Geometry/Excel correspondence is not a bijection')
    return binding, 'ORIGINAL_OBJECT_IDS' if same_ids else 'UNIQUE_GEOMETRY_BIJECTION'


def native_export(folder, rows, points):
    candidates = []
    errors = []
    for path in sorted(folder.glob('*.cbf')):
        try:
            objects, _ = extract_cbf_weld_json(path)
            index = JsonReferenceIndex(objects)
            welds = []
            for parent in objects:
                segments = []
                for i, part in enumerate(index.line_children(parent)):
                    a, b = np.asarray(part['points_reference_mm'], float)
                    samples = np.linspace(a, b, int(np.ceil(part['length_mm'])) + 1)
                    segments.append(dict(index=i, kind='line', lengthMm=part['length_mm'], points=samples.tolist()))
                welds.append(dict(id=str(parent['ObjectId']), segments=segments))
            binding, method = bind_geometry(welds, rows)
            samples = np.concatenate([np.asarray(s['points']) for w in welds for s in w['segments']])
            forward = float(cKDTree(points).query(samples, workers=2)[0].max())
            reverse = float(cKDTree(samples).query(points, workers=2)[0].max())
            if max(forward, reverse) > .501:
                raise ValueError(f'Native geometry disagrees with authoritative weld PLY: {forward}, {reverse}')
            candidates.append((welds, {str(o['ObjectId']): o for o in objects}, path, binding, method))
        except (ValueError, KeyError, AssertionError) as exc:
            errors.append(dict(file=path.name, reason=str(exc)))
    if len(candidates) != 1:
        raise ValueError(f'Expected one PLY/Excel-verified native export; got {len(candidates)}; {errors}')
    return candidates[0]


def compile_geometry(welds, rows, binding, aux, points, modern):
    parents = []
    samples = []
    categories = collections.Counter()
    for weld in welds:
        source_id = str(weld['id'])
        pid = binding[source_id]
        row = rows[pid]
        type_text = str(row['WeldType (isVerticalWeldLine)'])
        if ('Vertical' in type_text) == ('Horizontal' in type_text):
            raise ValueError('Ambiguous Excel weld type')
        weld_type = 'vertical' if 'Vertical' in type_text else 'horizontal'
        children = []
        ends = []
        for i, segment in enumerate(weld['segments']):
            p = np.asarray(segment['points'], float)
            if segment['index'] != i or p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
                raise ValueError('Invalid ordered source child')
            declared = float(segment['lengthMm'])
            if segment['kind'] == 'line':
                delta = p[-1] - p[0]
                length = float(np.linalg.norm(delta))
                if length <= 0 or abs(length - declared) >= ENDPOINT_TOLERANCE_MM:
                    raise ValueError('Degenerate line or inconsistent child length')
                if np.linalg.norm(np.cross(p - p[0], delta / length), axis=1).max() >= 1e-5:
                    raise ValueError('Source line samples are not collinear')
                rep = dict(kind='line', endpoints_mm=p[[0, -1]].tolist())
            elif segment['kind'] == 'arc':
                arc = fit_ordered_circle(p, tolerance_mm=1e-5, is_closed=np.linalg.norm(p[0] - p[-1]) < 1e-6)
                if abs(arc.length_mm - declared) >= LENGTH_TOLERANCE_MM:
                    raise ValueError('Inconsistent finite arc length')
                rep = dict(kind='arc', arc_parameters=arc.to_parameters())
            else:
                raise ValueError('Unsupported curve kind')
            samples.append(p)
            ends.append(p[[0, -1]])
            children.append(dict(original_parent_id=pid, original_child_id=pid + ':' + str(i),
                                 original_segment_index=i, original_kind=segment['kind'], weld_type=weld_type,
                                 representation=rep, original_attributes={}, attributes_status='SEPARATE_MASKED_EXCEL_EVIDENCE'))
            categories[weld_type + '/' + segment['kind']] += 1
        geometric_closed = np.linalg.norm(ends[0][0] - ends[-1][1]) <= 1e-6
        flag = aux.get(source_id, aux.get(pid, {})).get('IsClosedWeldLine')
        if geometric_closed and flag is None:
            raise ValueError('Closed parent semantics unresolved')
        closed = bool(flag) if flag is not None else False
        if closed and not geometric_closed:
            raise ValueError('Closed flag contradicts ordered geometry')
        gaps = ['SHARED' if np.linalg.norm(a[1] - b[0]) <= 1e-6 else 'POSITIVE_GAP' for a, b in zip(ends[:-1], ends[1:])]
        parents.append(dict(original_parent_id=pid, original_child_count=len(children), children=children,
                            gaps=gaps, is_closed=closed, source_geometry_parent_id=source_id))
    samples = np.concatenate(samples)
    forward = float(cKDTree(points).query(samples, workers=2)[0].max())
    reverse = float(cKDTree(samples).query(points, workers=2)[0].max())
    tolerance = 1e-5 if modern else .501
    if max(forward, reverse) >= tolerance:
        raise ValueError(f'Bidirectional PLY geometry disagreement: {forward}, {reverse}, tolerance {tolerance}')
    return dict(coordinate_frame='workpiece_local', unit='mm', parents=parents,
                original_parents=len(parents), original_children=sum(len(p['children']) for p in parents),
                unresolved_children=0), dict(categories=dict(categories), json_to_ply_max_mm=forward,
                                            ply_to_json_max_mm=reverse, sampling_validation_tolerance_mm=tolerance)


def attribute_evidence(target, rows):
    parents = []
    state, values, required, records = [], [], [], []
    counts = collections.Counter()
    for parent in target['parents']:
        row = rows[parent['original_parent_id']]
        children = []
        if len(parent['children']) != int(row['SegmentCount']):
            raise ValueError('Attribute/geometry child count disagreement')
        for i, child in enumerate(parent['children']):
            attributes, frames = {}, {}
            for side, outer_index in [('start', 0), ('end', len(parent['children']) - 1)]:
                outer = i == outer_index
                attributes[side] = {}
                for field, column in FIELDS[side].items():
                    label = numeric(row.get(column), field) if outer else dict(known=False, reason='NO_CHILD_ENDPOINT_EXPORT')
                    needed = child['weld_type'] == 'horizontal' or (side == 'end' and field == 'blocker_distance')
                    label.update(required_by_current_customer_rule=needed, source_column=column,
                                 source_scope='PARENT_OUTER_ENDPOINT' if outer else 'UNAVAILABLE')
                    attributes[side][field] = label
                    counts['known_required' if label['known'] and needed else 'unknown_required' if needed else 'optional'] += 1
                if outer:
                    prefix = side.title() + 'Matrix'
                    translation = row_endpoint(row, side.title())
                    error = float(np.linalg.norm(translation - endpoint(child, side)))
                    if error >= ENDPOINT_TOLERANCE_MM:
                        raise ValueError('Exported frame translation contradicts child endpoint')
                    axes = [[row.get(prefix + '.UpperLeft.' + a + '.' + b) for b in 'XYZ'] for a in 'XYZ']
                    matrix = np.asarray(axes, float)
                    finite = bool(np.isfinite(matrix).all())
                    orth = float(np.max(abs(matrix @ matrix.T - np.eye(3)))) if finite else None
                    determinant = float(np.linalg.det(matrix)) if finite else None
                    valid = bool(finite and orth < 1e-6 and abs(determinant - 1) < 1e-6)
                    frames[side] = dict(raw_export_axes=axes, translation=translation.tolist(), orthonormal=valid,
                                        known=valid, max_orthogonality_error=orth, determinant=determinant,
                                        axis_sign_not_reinterpreted=True, plate_normals_not_inferred=True)
                    counts['valid_frame' if valid else 'invalid_frame'] += 1
                else:
                    frames[side] = dict(known=False, reason='NO_INTERIOR_CHILD_FRAME_EXPORT')
            children.append(dict(child_id=child['original_child_id'], index=i, weld_type=child['weld_type'],
                                 endpoint_attributes=attributes, endpoint_frames=frames))
            st, val, req = [], [], []
            for side, field, name in ATTRIBUTE_FIELDS:
                label = attributes[side][field]
                code = ENCODING[label['encoding']] if label['known'] else -1
                st.append(code)
                val.append(float(label['raw_value']) if code == 0 else np.nan)
                req.append(label['required_by_current_customer_rule'])
            state.append(st)
            values.append(val)
            required.append(req)
            records.append(dict(parent_id=parent['original_parent_id'], child_id=child['original_child_id'],
                                child_index=i, weld_type=child['weld_type']))
        flags = {k: row.get(k) for k in ('HasEave (Any)', 'IsWorkRangeSplit (Any)', 'WorkRangeFrontSplit',
                                       'WorkRangeBackSplit', 'IsSplitedForDualRobot (Any)', 'HasMid (Any)')}
        parents.append(dict(parent_id=parent['original_parent_id'], raw_parent_scope_flags=flags,
                            parent_flags_not_broadcast=True, children=children))
    state = np.asarray(state, np.int8).reshape(-1, 8)
    values = np.asarray(values, float).reshape(-1, 8)
    required = np.asarray(required, bool).reshape(-1, 8)
    arrays = dict(state=state, exact_mm=values, required_mask=required,
                  state_mask=required & (state >= 0), numeric_mask=required & (state == 0))
    if not np.isfinite(values[arrays['numeric_mask']]).all() or not np.isnan(values[state != 0]).all():
        raise ValueError('Attribute unknown/exact encoding validation failed')
    return dict(parents=parents), arrays, records, dict(counts)


def qualify(example, manifest_sha):
    folder = DATA / example['directory']
    books, weld_plys, surfaces = [list(folder.glob(p)) for p in ('*.xlsx', '*weld_welds.ply', '*weld_workpiece.ply')]
    if any(len(files) != 1 for files in (books, weld_plys, surfaces)):
        raise ValueError('Expected one Excel, weld PLY, and workpiece PLY')
    book, weld_ply, surface = books[0], weld_plys[0], surfaces[0]
    header, point_count = ply_header(surface)
    rows = workbook_rows(book)
    points = ply_points(weld_ply)
    jsons = list(folder.glob('*weld_welds.json'))
    sources = [book, weld_ply, surface]
    if len(jsons) == 1:
        export = read(jsons[0])
        if export['unit'] != 'millimeter' or export['coordinateFrame'] != 'reference_workpiece':
            raise ValueError('Unconfirmed source geometry unit/frame')
        welds = export['welds']
        binding, method = bind_geometry(welds, rows)
        aux = {}
        cbfs = list(folder.glob('*.cbf'))
        if len(cbfs) == 1:
            objects, _ = extract_cbf_weld_json(cbfs[0])
            aux = {str(o['ObjectId']): o for o in objects}
            sources.append(cbfs[0])
        target, audits = compile_geometry(welds, rows, binding, aux, points, modern=True)
        sources.append(jsons[0])
        geometry_source = 'MODERN_ORDERED_GEOMETRY_VERIFIED_BY_PLY_AND_EXCEL'
    elif not jsons:
        # Preserve the already verified curved legacy label; never import the
        # old rebuilding script, whose top-level code mutates all five cases.
        prior = ROOT / 'server_rebuild/labels_v1/BM79A_native_local'
        if example['directory'] == '9月20日/BM79A':
            receipt = read(prior / 'receipt.json')
            if digest(prior / 'targets.json') != receipt['targets_sha256']:
                raise ValueError('Authenticated native curved label changed')
            for name, sha in receipt['files'].items():
                if digest(DATA / name) != sha:
                    raise ValueError('Native curved label source changed')
                if DATA / name not in sources:
                    sources.append(DATA / name)
            target = copy.deepcopy(read(prior / 'targets.json'))
            method = 'AUTHENTICATED_NATIVE_CURVE_REUSE'
            audits = dict(label_roundtrip_TP=receipt['label_roundtrip_TP'], prior_receipt_sha256=digest(prior / 'receipt.json'))
            geometry_source = method
        else:
            welds, aux, cbf, binding, method = native_export(folder, rows, points)
            target, audits = compile_geometry(welds, rows, binding, aux, points, modern=False)
            sources.append(cbf)
            geometry_source = 'AUXILIARY_NATIVE_BOUNDARIES_VERIFIED_BY_PLY_AND_EXCEL'
    else:
        raise ValueError('Ambiguous modern geometry files')
    target['role'] = example['split'].upper()
    evidence, arrays, records, attr_counts = attribute_evidence(target, rows)
    destination = BASE / 'labels' / example['key']
    destination.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=example['key'] + '_', dir=destination.parent) as temporary:
        temp = Path(temporary) / 'result'
        temp.mkdir()
        write(temp / 'targets.json', target)
        write(temp / 'endpoint_evidence.json', evidence)
        write(temp / 'attribute_records.json', records)
        np.savez(temp / 'attribute_labels.npz', **arrays)
        receipt = dict(key=example['key'], family_key=example['family_key'], family_id=example['family_id'],
                       directory=example['directory'], split=example['split'], role=target['role'],
                       status='QUALIFIED_GEOMETRY_WITH_MASKED_ATTRIBUTES', parents=target['original_parents'],
                       children=target['original_children'], workpiece_points=point_count,
                       workpiece_source=str(surface.relative_to(DATA)), geometry_source=geometry_source,
                       parent_binding_method=method, geometry_audit=audits, attribute_counts=attr_counts,
                       required_attributes_complete=attr_counts.get('unknown_required', 0) == 0,
                       split_manifest_sha256=manifest_sha, source_sha256=digest(__file__), fit_calls=0,
                       source_files={str(p.relative_to(DATA)): digest(p) for p in sources},
                       artifacts={p.name: digest(p) for p in temp.iterdir()})
        write(temp / 'receipt.json', receipt)
        os.replace(temp, destination)
    return receipt


def main(resume=False):
    manifest_path = BASE / 'split_manifest.json'
    manifest = read(manifest_path)
    manifest_sha = digest(manifest_path)
    results, failures = [], []
    started = time.perf_counter()
    examples = sorted(manifest['examples'], key=lambda e: (e['split'] != 'train', not e['canonical_version'], e['key']))
    for i, example in enumerate(examples):
        destination = BASE / 'labels' / example['key'] / 'receipt.json'
        try:
            if destination.exists():
                if not resume:
                    raise ValueError('Existing qualification requires explicit resume')
                receipt = read(destination)
                if receipt['split_manifest_sha256'] != manifest_sha or receipt['source_sha256'] != digest(__file__):
                    raise ValueError('Qualification binding changed')
                for name, sha in receipt['source_files'].items():
                    if digest(DATA / name) != sha:
                        raise ValueError('Qualified source changed')
                for name, sha in receipt['artifacts'].items():
                    if digest(destination.parent / name) != sha:
                        raise ValueError('Qualified artifact changed')
            else:
                receipt = qualify(example, manifest_sha)
            results.append(receipt)
            print('QUALIFIED', i + 1, '/', len(examples), example['split'], example['key'], receipt['children'], receipt['parent_binding_method'], flush=True)
        except Exception as exc:
            failure = dict(key=example['key'], family_id=example['family_id'], directory=example['directory'],
                           split=example['split'], type=type(exc).__name__, reason=str(exc))
            failures.append(failure)
            print('QUARANTINED', failure, flush=True)
        write(BASE / 'qualification_progress.json', dict(processed=i + 1, total=len(examples),
              qualified=len(results), quarantined=len(failures), seconds=time.perf_counter() - started))
        gc.collect()
    known = {r['key'] for r in results} | {f['key'] for f in failures}
    if known != {e['key'] for e in examples} or not results:
        raise ValueError('Incomplete or entirely unsuccessful qualification')
    summary = dict(status='ALL_VERSIONS_ACCOUNTED_WITH_EXPLICIT_QUARANTINES', qualified=results, quarantined=failures,
                   qualified_directories_by_split=dict(collections.Counter(r['split'] for r in results)),
                   qualified_families_by_split={s: len({r['family_id'] for r in results if r['split'] == s})
                                                for s in ('train', 'validation', 'test')},
                   qualified_children_by_split={s: sum(r['children'] for r in results if r['split'] == s)
                                                for s in ('train', 'validation', 'test')},
                   all_directories_accounted=True, split_manifest_sha256=manifest_sha, source_sha256=digest(__file__),
                   test_model_predictions=0, fit_calls=0, seconds=time.perf_counter() - started)
    write(BASE / 'qualification_summary.json', summary)
    print('QUALIFICATION_COMPLETE', summary['qualified_directories_by_split'], summary['qualified_families_by_split'], 'quarantined', len(failures), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args.resume)
