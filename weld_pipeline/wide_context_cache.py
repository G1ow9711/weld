"""Immutable sidecar arrays over the existing verified curved sequence cache."""
import json
import os
from pathlib import Path
import numpy as np
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece as load_base
from weld_pipeline.wide_vertical_features import ContextFeatures, EXTRA_CHANNELS, CHANNELS, FEATURE_VERSION

STORAGE_FORMAT = 'wide_vertical_context_overlay_v1'


def write_piece(folder, base_folder, extras, provenance):
    folder, base_folder = Path(folder).resolve(), Path(base_folder).resolve()
    base = load_base(base_folder)
    if len(extras) != len(base['sequences']):
        raise ValueError('Context sequence count differs')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        old = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if old['provenance'] != provenance or old['base_manifest_sha256'] != sha256(base_folder / 'manifest.json'):
            raise ValueError('Frozen overlay differs')
        load_piece(folder)
        return old
    entries, arrays, offset = [], [], 0
    for index, (sequence, extra) in enumerate(zip(base['sequences'], extras)):
        if sequence['type'] == 'vertical':
            if extra is None or extra.shape != (len(sequence['xyz']), EXTRA_CHANNELS) or extra.dtype != np.float32:
                raise ValueError('Missing or invalid vertical context')
            if not np.isfinite(extra).all() or (extra < 0.).any() or (extra > 1.).any():
                raise ValueError('Context outside normalized distance range')
            entries.append({'sequence': index, 'start': offset, 'stop': offset + len(extra)})
            offset += len(extra)
            arrays.append(extra)
        elif extra is not None:
            raise ValueError('Horizontal context must be implicit zeros')
    values = np.concatenate(arrays) if arrays else np.empty((0, EXTRA_CHANNELS), np.float32)
    np.save(folder / 'vertical_context.npy', values, allow_pickle=False)
    result = {'storage_format': STORAGE_FORMAT, 'feature_version': FEATURE_VERSION,
              'base_folder': os.path.relpath(base_folder, folder),
              'base_manifest_sha256': sha256(base_folder / 'manifest.json'),
              'sequence_count': len(base['sequences']), 'vertical_sequences': entries,
              'context_shape': list(values.shape), 'context_sha256': sha256(folder / 'vertical_context.npy'),
              'provenance': provenance, 'label_policy': 'Reuse original target and truth arrays unchanged, including all-channel -1 unknown rows.'}
    temporary = folder / 'manifest.pending.json'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(folder / 'manifest.json')
    load_piece(folder)
    return result


def load_piece(folder, verify=True):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['storage_format'] != STORAGE_FORMAT or manifest['feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported context cache')
    base_folder = (folder / manifest['base_folder']).resolve()
    if not base_folder.is_relative_to(Path.cwd().resolve()):
        raise ValueError('Base cache outside project')
    if sha256(base_folder / 'manifest.json') != manifest['base_manifest_sha256']:
        raise ValueError('Base cache manifest changed')
    base = load_base(base_folder, verify=verify)
    path = folder / 'vertical_context.npy'
    if verify and sha256(path) != manifest['context_sha256']:
        raise ValueError('Context array changed')
    context = np.load(path, mmap_mode='r' if manifest['context_shape'][0] else None, allow_pickle=False)
    if list(context.shape) != manifest['context_shape'] or context.dtype != np.float32:
        raise ValueError('Context shape or dtype changed')
    if len(base['sequences']) != manifest['sequence_count']:
        raise ValueError('Base sequence count changed')
    indices, previous = {}, 0
    for entry in manifest['vertical_sequences']:
        index, start, stop = entry['sequence'], entry['start'], entry['stop']
        if index in indices or not 0 <= index < len(base['sequences']) or start != previous or stop <= start:
            raise ValueError('Invalid context bounds')
        sequence = base['sequences'][index]
        if sequence['type'] != 'vertical' or stop - start != len(sequence['xyz']):
            raise ValueError('Context does not match vertical sequence')
        indices[index] = (start, stop)
        previous = stop
    if previous != len(context):
        raise ValueError('Unreferenced context rows')
    sequences = []
    for index, sequence in enumerate(base['sequences']):
        if (sequence['type'] == 'vertical') != (index in indices):
            raise ValueError('Missing vertical context')
        extra = context[slice(*indices[index])] if index in indices else None
        sequences.append({**sequence, 'features': ContextFeatures(sequence['features'], extra)})
    return {**base, 'sequences': sequences,
            'record': {**base['record'], 'feature_version': FEATURE_VERSION, 'feature_channels': CHANNELS},
            'cache_manifest_sha256': sha256(folder / 'manifest.json')}
