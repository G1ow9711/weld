"""TRAIN-only geometry carrier scope audit from previously frozen labels."""
from collections import Counter
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc


def main():
    data = OUT / 'run186';proposals = OUT / 'run185'
    manifest = read(data / 'training_manifest.json')
    rows = []
    for case in manifest['cases']:
        key = case['source_key'];assert key not in ('64', '24')
        folder = proposals / ('source_' + key)
        freeze = read(folder / 'input_freeze.json')
        assert digest(folder / 'profiles.json') == freeze['files']['profiles.json']
        profiles = read(folder / 'profiles.json')
        supervision = read(data / ('source_' + key + '_supervision.json'))
        branches = [p['proposal_branch'] for p in profiles]
        assert digest(data / case['input_file']) == case['input_sha256']
        with np.load(data / case['input_file'], allow_pickle=False) as z:
            visited = set(z['nodes'].tolist())
        compatible_ids = {int(i) for i in supervision['by_profile']}
        counts = Counter(branches)
        compatible = Counter(branches[int(i)] for i in supervision['by_profile'])
        coverage = Counter();native_covered = 0;native_only = 0
        old_only = 0;child_rows = []
        for child in supervision['children']:
            available = {branches[i] for i in child['profiles']}
            coverage.update(available)
            has_native = any(b.startswith('native_learned_finite_') for b in available)
            has_old = any(not b.startswith('native_learned_finite_') for b in available)
            native_covered += has_native;native_only += has_native and not has_old
            old_only += has_old and not has_native
            child_rows.append(dict(parent=child['parent'], child=child['child'], kind=child['kind'],
                                   supported_branches=sorted(available)))
        branch_rows = [dict(branch=b, carriers=counts[b], compatible_carriers=compatible[b],
                            supported_G=coverage[b], background_carriers=counts[b]-compatible[b]) for b in sorted(counts)]
        assert sum(r['carriers'] for r in branch_rows) == len(profiles)
        assert len(child_rows) + len(supervision['missing']) == case['original_G']
        row = dict(source_key=key, original_G=case['original_G'], native_finite_supported_G=native_covered,
                   native_only_G=native_only, old_only_G=old_only, branches=branch_rows,
                   carrier_count=len(profiles), compatible_carrier_count=sum(compatible.values()),
                   actual_run187_queried_carriers=len(visited),
                   actual_run187_queried_compatible_carriers=len(visited & compatible_ids),
                   actual_run187_queried_background_carriers=len(visited - compatible_ids),
                   background_carrier_count=len(profiles)-len(compatible_ids),
                   eight_epochs_repeat_same_query_carriers=True,
                   unseen_carrier_count_is_exposure_only_not_proof_of_unlearnability=True,
                   children=child_rows, original_denominator_preserved=True)
        rows.append(row)
        print('TRAIN_CARRIER_SCOPE', key, 'NATIVE_SUPPORT', native_covered, '/', case['original_G'],
              'CARRIERS', len(profiles), 'COMPATIBLE', sum(compatible.values()), flush=True)
        del profiles, supervision;gc.collect()
    write(OUT / 'run190/TRAIN_carrier_scope_audit.json',
          dict(rows=rows, original_training_G=sum(r['original_G'] for r in rows),
               native_finite_supported_G=sum(r['native_finite_supported_G'] for r in rows),
               original_eight_G=496, supplemental_TRAIN_G=4,
               not_extraction_accuracy=True, no_candidate_removed=True, no_heldout_target_read=True,
               optimizer_updates=0, source_sha256=digest(__file__), end_utc=utc()))


if __name__ == '__main__':
    main()
