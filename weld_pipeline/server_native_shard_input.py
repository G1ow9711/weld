"""Genuine original native-forest inputs from a complete compressed shard.

Surface normals and request context must be preserved original artifacts.
Their absence is an explicit error. No estimated surfaces, padded features,
reference geometry or filename identities are supplied to a model.
"""
import gzip
import hashlib
import json
from pathlib import Path
import pickle

from .sjg_joint_segment_graph_1104 import JointSegmentGraph
from .sjg_local_runtime_904 import digest, read


def original_json(path, expected_sha):
    with gzip.open(path, 'rb') as stream:
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('Preserved input context differs from its original bytes')
    return json.loads(data)


def load_native_input(dataset, job_key, context_root):
    inp = dataset.input_case(job_key)
    job = dataset.jobs[job_key]
    folder = Path(context_root) / job_key
    receipt = read(folder / 'receipt.json')
    shard_receipt = read(dataset.shards / job_key / 'receipt.json')
    if (receipt['job_key'] != job_key or receipt['runtime_key'] != job['runtime_key']
            or receipt['registry_sha256'] != dataset.registry_sha256
            or receipt['workpiece_sha256'] != job['workpiece_sha256']
            or receipt['GT_geometry_reads'] != 0 or receipt['STEP_reads'] != 0
            or receipt['input_metadata_fabricated']
            or receipt['original_geometry_sha256'] != inp.bindings['geometry_sha256']
            or receipt['original_profiles_sha256'] != shard_receipt['files']['profiles.json']):
        raise ValueError('Native context is not bound to this complete input shard')
    for name, sha in receipt['files'].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError('Genuine native input context changed')
    binding = read(folder / 'source_bindings.json')
    surfaces = original_json(folder / 'surfaces.json.gz', binding['surface_bytes_sha256'])
    requests = original_json(folder / 'input_requests.json.gz', binding['request_bytes_sha256'])
    # Match the original bounded geometry loader's retained fields exactly.
    profiles = []
    for original in inp.profiles:
        profile = dict(representation=original['representation'], face_indices=original.get('face_indices', []))
        profile.update({name: original[name] for name in ('support_intervals_mm', 'unobserved_gaps_mm')
                        if name in original})
        profiles.append(profile)
    graph = JointSegmentGraph(profiles, inp.offsets, inp.spans)
    return dict(key=job['runtime_key'], x=inp.raw, canonical=inp.canonical, graph=graph,
        surfaces=surfaces, frame_cache={}, input_requests=requests,
        feature_receipt_sha256=digest(dataset.shards / job_key / 'receipt.json'),
        feature_receipt_is_full_shard_provenance_not_original_receipt=True,
        alias_map_sha256=inp.bindings['alias_sha256'], geometry_sha256=inp.bindings['geometry_sha256'],
        preserved_context_receipt_sha256=digest(folder / 'receipt.json'),
        GT_geometry_reads=0, STEP_reads=0)


def load_frozen_native_models(root):
    """Load the native object/endpoint/connection weights underlying run446."""
    root = Path(root)
    reference = read(root / 'experiment_records/run446_frozen_baseline_reference_20261011.json')
    binding = next(row for row in reference['checkpoints'] if row['path'].endswith('/run379/model.pkl'))
    path = root / binding['path']
    if digest(path) != binding['sha256']:
        raise ValueError('Original best native upstream model fingerprint changed')
    with path.open('rb') as stream:
        models = pickle.load(stream)
    if (models['object'].n_features_in_ != 152 or models['endpoint'].n_features_in_ != 153
            or models['connection'].n_features_in_ != 32):
        raise ValueError('Original best native feature contracts changed')
    # The original loader disables console verbosity without altering weights.
    for name in ('object', 'endpoint', 'connection'):
        models[name].verbose = 0
    return models, binding
