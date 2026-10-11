"""Immutable local experiment records for Git, including checkpoint fingerprints.

Large checkpoints and generated arrays stay outside Git. Records bind the
source revision, public source snapshot, split, configuration and result files.
Record contents are private local development history, not auto-publication.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'experiment_records'


def fingerprint(name):
    path = (ROOT / name).resolve()
    path.relative_to(ROOT)
    if any(part in {'.git', '.codex', '.aws', 'github_auth', 'profile'} for part in path.parts):
        raise ValueError('Account state and credentials cannot be experiment evidence')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)


def git_head(folder):
    result = subprocess.run(['git', '-C', str(folder), 'rev-parse', '--verify', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def record(record_id, status, evidence, checkpoints, metadata):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,95}', record_id):
        raise ValueError('Invalid experiment record ID')
    DEST.mkdir(exist_ok=True)
    path = DEST / (record_id + '.json')
    snapshots = DEST / 'evidence' / record_id
    if path.exists() or snapshots.exists():
        raise ValueError('Experiment records are immutable; use a new revision ID')
    evidence_records = [fingerprint(p) for p in evidence]
    if sum(item['bytes'] for item in evidence_records) > 32 * 1024**2:
        raise ValueError('Experiment evidence exceeds the 32MiB snapshot limit')
    for item in evidence_records:
        if Path(item['path']).suffix not in {'.json', '.md', '.txt'} or item['bytes'] > 8 * 1024**2:
            raise ValueError('Snapshot evidence must be a small JSON/text document; weights use --checkpoint')
    value = dict(record_id=record_id, status=status, recorded_utc=datetime.now(timezone.utc).isoformat(),
                 source_commit_before_record=git_head(ROOT),
                 public_source_commit=git_head(ROOT / '.codex-local/public_weld'),
                 evidence=evidence_records, checkpoints=[fingerprint(p) for p in checkpoints],
                 metadata=metadata, checkpoints_embedded_in_git=False, automatic_publication=False,
                 recording_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    snapshots.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=record_id + '-', dir=snapshots.parent))
    try:
        for index, item in enumerate(evidence_records):
            name = f'{index:03d}_{Path(item["path"]).name}'
            content = (ROOT / item['path']).read_bytes()
            if hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Evidence changed during recording')
            (temporary / name).write_bytes(content)
            item['snapshot_path'] = str((snapshots / name).relative_to(ROOT))
        temporary.rename(snapshots)
        with path.open('x') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print('EXPERIMENT_RECORD_WRITTEN', path.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--evidence', action='append', default=[])
    parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--metadata-file')
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata_file).read_text()) if args.metadata_file else {}
    record(args.id, args.status, args.evidence, args.checkpoint, metadata)
