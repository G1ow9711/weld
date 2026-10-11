"""Reconstruct a frozen parent pool from declared input prediction artifacts.

No reference labels, supervision mask, or source-key lookup is used. Paths and
hashes bind the candidate artifacts; geometry is not fitted or corrected here.
"""
from pathlib import Path
import numpy as np
from .sjg_local_runtime_904 import read,digest
from .server_compact_line_predictions import decode_compact_lines


def load_parent_pool(root, manifest):
    root=Path(root)
    original=root/manifest['native_parents']['path']
    compact_path=root/manifest['line_predictions']['path']
    if digest(original)!=manifest['native_parents']['sha256'] or digest(compact_path)!=manifest['line_predictions']['sha256']:
        raise ValueError('Changed parent-pool input artifact')
    native=read(original)
    with np.load(compact_path) as z:compact={name:z[name] for name in z.files}
    added=decode_compact_lines(compact)
    for parent in added['parents']:
        if parent['child_count']!=1:raise ValueError('Unexpected compact line structure')
        # This is the actual input carrier that produced the interval, not a
        # geometry-based assignment to a reference or nearest target.
        parent['segments'][0]['profile']=int(parent['input_profile'])
    parents=native['parents']+added['parents']
    invalid=native['invalid_predictions']+added['invalid_predictions']
    total=sum(p['child_count'] for p in parents)+sum(v['counted_false_children'] for v in invalid)
    if total!=native['total_prediction_children']+added['total_prediction_children']:
        raise ValueError('Candidate child ledger mismatch')
    return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=total),len(native['parents'])
