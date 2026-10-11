"""Verify cached inference dependencies and retain historical entrypoint evidence."""
from pathlib import Path
from training.labels import sha256


def verify_cached_inference_sources(state, checkpoint):
    # Neither entrypoint is imported by the cached calibration/evaluation path.
    historical_entrypoints = {'train_typed_pilot.py', 'predict_typed.py'}
    archived = []
    for name, wanted in state['source_hashes'].items():
        source = Path(name)
        if source.exists() and sha256(source) == wanted:
            continue
        snapshot = Path(checkpoint).parent / 'source_snapshot' / source.name
        if source.name not in historical_entrypoints:
            raise ValueError('Cached inference dependency changed: ' + name)
        if not snapshot.exists() or sha256(snapshot) != wanted:
            raise ValueError('Historical entrypoint evidence missing: ' + name)
        archived.append({'source': name, 'snapshot': str(snapshot), 'sha256': wanted,
                         'reason': 'Historical entrypoint is not executed by cached inference.'})
    return archived
