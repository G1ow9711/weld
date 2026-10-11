"""Losslessly bundle a completed TRAIN export and retain its admission records.

Only the named, completed generated CSV control is eligible. Every archive
member is read back and matched against the original file SHA before removing
its redundant raw copy. Input clouds, predictions and model weights are not
part of this operation. The persistent archive can restore the exact export.
"""
import argparse
import hashlib
from pathlib import Path
import tarfile

from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'server_rebuild/full_dataset_v1'


def main(args):
    if not args.experiment_name or args.experiment_name in ('.', '..') or Path(args.experiment_name).name != args.experiment_name:
        raise ValueError('A named completed export required')
    folder = BASE / args.experiment_name
    completion = read(folder / 'receipt.json')
    if (completion['status'] != 'ALL_AUTHENTICATED_FROZEN_OUTPUTS_CSV_EXPORTED_NOT_CERTIFIED'
            or completion['role'] != 'TRAIN' or completion['fit_calls'] != 0
            or completion['reference_geometry_reads'] != 0
            or completion['protocol_sha256'] != digest(folder / 'protocol.json')):
        raise ValueError('Only a complete authenticated TRAIN export control may be archived')
    files = {}
    redundant = []
    for item in completion['records']:
        receipt = (ROOT / item['path']).resolve(); receipt.relative_to(folder)
        if digest(receipt) != item['sha256']:
            raise ValueError('Completed export record changed')
        files[str(receipt.relative_to(folder))] = item['sha256']
        for name, sha in read(receipt)['artifacts'].items():
            path = (receipt.parent / name).resolve(); path.relative_to(receipt.parent)
            if path.is_symlink() or not path.is_file() or digest(path) != sha:
                raise ValueError('Completed CSV artifact changed')
            files[str(path.relative_to(folder))] = sha
            redundant.append(path)
    for name in ('protocol.json', 'receipt.json'):
        files[name] = digest(folder / name)
    if set(files) != {str(path.relative_to(folder)) for path in folder.rglob('*') if path.is_file()}:
        raise ValueError('Archive would omit an existing export file')
    archive = folder.with_suffix('.tar.gz')
    if archive.exists() or (folder / 'archive_storage_binding.json').exists():
        raise ValueError('Export archival is immutable and must not replace an earlier bundle')
    original_bytes = sum((folder / name).stat().st_size for name in files)
    with tarfile.open(archive, 'x:gz', compresslevel=1) as stream:
        for index, name in enumerate(sorted(files)):
            path = folder / name
            if digest(path) != files[name]:
                raise ValueError('Export changed while bundling')
            stream.add(path, arcname=name, recursive=False)
            if index % 8192 == 0:
                print('CSV_BUNDLE_WRITTEN', index, '/', len(files), flush=True)
    checked = set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream:
            if not member.isfile() or member.name not in files or member.name in checked:
                raise ValueError('Unexpected, repeated or unsafe archive member')
            checksum = hashlib.sha256()
            with stream.extractfile(member) as contents:
                while block := contents.read(4 * 1024**2):
                    checksum.update(block)
            if checksum.hexdigest() != files[member.name]:
                raise ValueError('Persistent bundle changed an original file')
            checked.add(member.name)
    if checked != set(files):
        raise ValueError('Persistent bundle lost an original file')
    binding = dict(status='COMPLETE_TRAIN_CSV_BUNDLE_ALL_ORIGINAL_BYTES_VERIFIED',
        source_sha256=digest(Path(__file__)), archive_path=str(archive.relative_to(ROOT)),
        archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
        original_bytes=original_bytes, original_files=len(files),
        original_export_receipt_sha256=digest(folder / 'receipt.json'),
        original_export_protocol_sha256=digest(folder / 'protocol.json'),
        all_members_read_back_and_sha_verified=True,
        raw_artifacts_removed_only_after_persistent_bundle_verification=True,
        restore='Extract bundle into a new directory; CSV curve paths are relative to that directory',
        input_clouds_predictions_and_weights_changed=False)
    write(folder / 'archive_storage_binding.json', binding)
    # Keep top-level protocol/receipt and every admission record. All removed
    # files are generated artifacts whose exact bytes remain in the bundle.
    for path in redundant:
        expected = files[str(path.relative_to(folder))]
        if digest(path) != expected:
            raise ValueError('Raw file changed after archive verification; retain it')
        path.unlink()
    print('COMPLETE_TRAIN_CSV_BUNDLE_VERIFIED', original_bytes, archive.stat().st_size, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
