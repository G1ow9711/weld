"""Refuse new fitting on auxiliary models with confirmed supervision conflicts.

Frozen inference and historical evaluation remain available. A repaired model
uses a new output directory with its own independently audited labels.
"""
from weld_pipeline.sjg_local_runtime_904 import OUT, read, digest


def require_clean_occupancy_supervision(occupancy_runs):
    for run in sorted(set(occupancy_runs)):
        marker = OUT / run / 'confirmed_supervision_conflicts.json'
        if not marker.exists():
            continue
        item = read(marker)
        evidence = OUT / item['evidence_file']
        if digest(evidence) != item['evidence_sha256']:
            raise RuntimeError(f'Auxiliary supervision conflict evidence changed: {run}')
        audit = read(evidence)
        if audit['total_false_background_sequences'] > 0:
            raise RuntimeError(
                f'{run} has confirmed false-background training sequences; '
                'rebuild and independently audit native occupancy supervision, '
                'fit a new auxiliary model, and regenerate its dependent features '
                'before further fitting. Frozen inference remains permitted.')
