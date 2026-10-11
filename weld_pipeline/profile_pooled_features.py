"""Profile point-only feature construction without loading any reference labels."""
import argparse
import cProfile
import io
import json
import pstats
from pathlib import Path
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.pooled_mixed_sequences import make_fast_mixed_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcd', required=True)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    xyz = read_xyz_pcd(Path(a.pcd)) * 1000
    profiler = cProfile.Profile()
    sequences, metadata = profiler.runcall(make_fast_mixed_sequences, xyz)
    profiler.dump_stats(str(out / 'features.prof'))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative')
    stats.print_stats(40)
    (out / 'profile.txt').write_text(stream.getvalue(), encoding='utf-8')
    report = {'input_sha256': sha256(a.pcd), 'sequence_count': len(sequences),
              'feature_points': sum(len(s['xyz']) for s in sequences),
              'scope': 'Instrumented feature diagnosis; may overlap training, not an acceptance timing.',
              'sources': {str(p): sha256(p) for p in [Path(__file__), Path('weld_pipeline/pooled_mixed_sequences.py')]},
              'geometry': metadata}
    (out / 'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(stream.getvalue())


if __name__ == '__main__':
    main()
