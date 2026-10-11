"""Receive the explicitly scoped original-backbone supplement without overwrites."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = 'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run162/'


def allowed(name):
    p = PurePosixPath(name)
    return (not p.is_absolute() and '..' not in p.parts and '\\' not in name
            and (name in {RUN + 'last.pt', RUN + 'receipt.json',
                          'training/__init__.py', 'training/third_party/__init__.py'}
                 or name.startswith('training/third_party/ptv3_801/')))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def receive(archive, expected):
    assert digest(archive.read_bytes()) == expected, 'Archive SHA mismatch'
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP members'
        rows = json.loads(z.read('manifest.json'))['files']
        assert len(rows) == len({r['path'] for r in rows}), 'Duplicate manifest paths'
        assert set(names) == {'manifest.json'} | {r['path'] for r in rows}
        assert {RUN + 'last.pt', RUN + 'receipt.json'} <= set(names)
        assert 'training/third_party/ptv3_801/model.py' in names
        # Validate every member and collision before writing any project file.
        for r in rows:
            name = r['path']; assert allowed(name), name
            target = ROOT / name
            assert target.resolve().is_relative_to(ROOT), 'Path escaped project'
            data = z.read(name)
            assert len(data) == r['bytes'] and digest(data) == r['sha256'], name
            if target.exists():
                assert target.is_file() and digest(target.read_bytes()) == r['sha256'], 'Refuse different existing file: ' + name
        receipt = json.loads(z.read(RUN + 'receipt.json'))
        assert receipt['checkpoint_sha256'] == digest(z.read(RUN + 'last.pt'))
        for r in rows:
            target = ROOT / r['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open('xb') as f:
                    f.write(z.read(r['path']))
            assert digest(target.read_bytes()) == r['sha256']
    result = dict(status='ORIGINAL_BACKBONE_SUPPLEMENT_VERIFIED', archive_sha256=expected,
                  files=rows, original_checkpoint_receipt_matched=True, inference_parity_proven=False)
    out = ROOT / 'server_rebuild/best_baseline_restore/supplement_receipt.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'files': len(rows), 'receipt': str(out)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--sha256', required=True)
    args = ap.parse_args()
    receive(args.archive, args.sha256)
