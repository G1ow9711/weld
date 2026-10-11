"""Archive only source files whose bytes match completed inference receipts."""
import argparse
import json
import shutil
from pathlib import Path
from training.labels import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--decoder-config', required=True)
    ap.add_argument('--equivalence-receipt', default='weld_pipeline/reports/fast_mixed_equivalence/receipt.json')
    a = ap.parse_args()
    out, root = Path(a.outdir), Path.cwd().resolve()
    inference = json.loads((out / 'inference_receipt.json').read_text())
    score = json.loads((out / 'independent_score.json').read_text())
    entries = []
    for category, sources in [('inference', inference['inference_source_hashes']),
                              ('evaluation', score['evaluation_source_hashes'])]:
        for source, wanted in sources.items():
            path = Path(source).resolve()
            if not path.is_relative_to(root) or sha256(path) != wanted:
                raise ValueError('Source outside project or differs from receipt: ' + str(path))
            entries.append((path, out / 'source_snapshot' / category / path.relative_to(root), wanted))
    config = Path(a.decoder_config)
    if sha256(config) != inference['decoder_configuration']['configuration_sha256']:
        raise ValueError('Decoder configuration changed')
    equivalence = Path(a.equivalence_receipt)
    if sha256(equivalence) != inference['feature_equivalence_receipt_sha256']:
        raise ValueError('Feature equivalence receipt changed')
    for source, dest, wanted in entries:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if sha256(dest) != wanted:
            raise ValueError('Archived source verification failed')
    shutil.copy2(config, out / 'source_snapshot/decoder_config.json')
    shutil.copy2(equivalence, out / 'source_snapshot/fast_feature_equivalence.json')
    report = {'status': 'PASS', 'source_files': len(entries),
              'sources': {str(dest.relative_to(out)): wanted for _, dest, wanted in entries},
              'archiver_sha256': sha256(__file__)}
    (out / 'source_snapshot/verified_sources.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'source_files': len(entries), 'outdir': str(out)}))


if __name__ == '__main__':
    main()
