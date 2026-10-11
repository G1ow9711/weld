"""Compare exact and rounded equivalent-frame collisions, without editing labels."""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.workpiece_relative_context import frame_views

BASE = Path('weld_pipeline/data/material_support_v1')
DATA = Path('weld_pipeline/data/workpiece_context_v1')
OUT = Path('weld_pipeline/reports/workpiece_context_collisions_153_20261001.json')


def main():
    if OUT.exists():
        raise ValueError('Preserve relative context collision evidence')
    cfg = json.loads((DATA/'config.json').read_text('utf8'))
    xs = []; contexts = []; labels = []; records = []
    for name in cfg['pieces']:
        manifest = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        ctx_manifest = json.loads((DATA/name/'manifest.json').read_text('utf8'))
        for root, files in ((BASE/name, manifest['files']), (DATA/name, ctx_manifest['files'])):
            for file, digest in files.items():
                if sha256(root/file) != digest:
                    raise ValueError('Fixed data changed')
        xs.append(np.load(BASE/name/'signed.npy', allow_pickle=False))
        contexts.append(np.load(DATA/name/'context.npy', allow_pickle=False))
        with np.load(BASE/name/'labels.npz', allow_pickle=False) as y:
            labels.append(y['business'])
        records.extend(dict(piece=name, sequence=r['sequence'], sample_index=r['sample_index'])
                       for r in manifest['entries'])
    local = np.concatenate(xs); ctx = np.concatenate(contexts); y = np.concatenate(labels)
    reps = {'baseline':local, 'gravity':np.c_[local, ctx[:, 19:22]], 'whole':np.c_[local, ctx]}
    results = {}
    for representation, x in reps.items():
        views = frame_views(x, representation); results[representation] = {}
        for decimals in (None, 6):
            vv = views if decimals is None else [np.round(v, decimals) for v in views]
            canonical = np.stack([min((v[i] for v in vv), key=lambda a:tuple(a)) for i in range(len(x))])
            # Signed zeros describe identical physics and must not create false unique groups.
            canonical[canonical == 0.] = 0.
            packed = np.ascontiguousarray(canonical).view(np.dtype((np.void, x.shape[1]*x.dtype.itemsize))).ravel()
            _, inverse = np.unique(packed, return_inverse=True)
            groups = {}
            for i, group in enumerate(inverse):
                if y[i] >= 0:
                    groups.setdefault(int(group), []).append(i)
            conflicts = []
            for indices in groups.values():
                values, counts = np.unique(y[indices], return_counts=True)
                if len(values) > 1:
                    conflicts.append(dict(minimum_sample_errors=int(counts.sum()-counts.max()),
                        rows=[dict(records[i], label=int(y[i])) for i in indices]))
            result = dict(unique_groups=len(groups), conflicting_groups=len(conflicts),
                minimum_unavoidable_sample_errors=sum(r['minimum_sample_errors'] for r in conflicts), conflicts=conflicts)
            results[representation]['exact' if decimals is None else 'rounded_6_decimals'] = result
            print('WORKPIECE_CONTEXT_COLLISIONS', representation, decimals, result['conflicting_groups'], flush=True)
    OUT.write_text(json.dumps(dict(rows=len(y), results=results,
        data_config_sha256=sha256(DATA/'config.json'), audit_sha256=sha256(__file__),
        interpretation='Descriptor collisions only. No claim of model or full-PCD capacity limits.',
        original_labels_changed=False, observed_check_read=False, full_contract_pass=False),
        indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
