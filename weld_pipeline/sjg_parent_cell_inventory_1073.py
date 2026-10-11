"""All-family input cells and exact-XYZ query budget audit, without labels."""
import argparse
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc, check_resources
from weld_pipeline.sjg_sparse_contact_domain_1048 import family_cells
from weld_pipeline.sjg_hierarchical_features_1029 import positions
from weld_pipeline.sjg_projected_boundary_anchors_1050 import ProjectedBoundaryAnchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    a = ap.parse_args()
    key = a.source
    dest = OUT / 'run252' / ('source_' + key)
    dest.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    limit = 400000
    try:
        check_resources()
        case = next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key'] == key)
        db = next(c for c in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases'] if c['source_key'] == key)
        base = OUT / f'run185/source_{key}'
        binding = read(base/'input_freeze.json')
        for name in ('profiles.json', 'surfaces.json', 'surface_support.npz'):
            assert digest(base/name) == binding['files'][name]
        for file, field in ((db['json_file'], 'json_sha256'), (db['mapping_file'], 'mapping_sha256')):
            assert digest(OUT/'run201'/file) == db[field]
        assert digest(OUT/case['input_file']) == case['input_sha256'] == binding['input_sha256']
        domain = read(OUT/'run201'/db['json_file'])
        profiles = read(base/'profiles.json')
        faces = read(base/'surfaces.json')
        with np.load(OUT/'run201'/db['mapping_file']) as z:
            transform = z['local_to_canonical']
        xyz = np.load(OUT/case['input_file'], mmap_mode='r')
        fb = read(OUT/f'run166/source_{key}_freeze.json')
        feature_file = OUT/'run166'/fb['feature_file']
        assert fb['input_sha256'] == case['input_sha256']
        assert digest(feature_file) == fb['feature_sha256']
        with np.load(feature_file) as z:
            boundary = z['boundary']
        offsets = [0]
        query_set = set()
        exceeded_at = None
        sequences = []
        with np.load(base/'surface_support.npz') as support, (dest/'cells.f64').open('wb') as stream:
            provider = ProjectedBoundaryAnchors(xyz, boundary, faces, support)
            for fid, family in enumerate(domain['families']):
                extra = provider.family(family, domain['relations'][family['relation']]['ordered_face_indices'])
                cells, chains, cyclic = family_cells(family, profiles, transform, extra)
                stream.write(cells.astype('<f8', copy=False).tobytes())
                sequences.extend(dict(family=fid, start=offsets[-1]+lo, end=offsets[-1]+hi,
                                      cyclic=cyclic) for lo, hi in chains)
                offsets.append(offsets[-1]+len(cells))
                # Cap only this diagnostic set after proving the budget is exceeded.
                # All subsequent cells and all candidate identities remain stored.
                if exceeded_at is None:
                    q = np.c_[cells[:,0], (cells[:,0]+cells[:,1])/2, cells[:,1]].ravel()
                    coords = positions(family, q)
                    coords[coords == 0] = 0
                    for point in coords:
                        query_set.add(tuple(float(v) for v in point))
                        if len(query_set) > limit:
                            exceeded_at = fid
                            break
                if fid % 512 == 0:
                    check_resources()
                    write(dest/'progress.json', dict(source=key, families_done=fid+1,
                        families_total=len(domain['families']), cells=offsets[-1],
                        exact_XYZ_query_count_or_lower_bound=len(query_set),
                        query_budget_exceeded=exceeded_at is not None, utc=utc()))
                    print('PARENT_INPUT_CELLS', key, fid+1, '/', len(domain['families']),
                          'CELLS', offsets[-1], 'XYZ_COUNT_OR_LOWER_BOUND', len(query_set), flush=True)
            boundary_meta = provider.metadata()
        offsets = np.asarray(offsets, np.int64)
        np.save(dest/'family_offsets.npy', offsets)
        write(dest/'sequences.json', sequences)
        if exceeded_at is None:
            np.save(dest/'query_xyz.npy', np.asarray(sorted(query_set), np.float64).reshape(-1, 3))
        receipt = dict(status='INPUT_CELL_DOMAIN_COMPLETE', source=key, original_profiles=len(profiles),
            all_families=len(domain['families']), all_relations=len(domain['relations']),
            cells=int(offsets[-1]), exact_XYZ_query_count_or_lower_bound=len(query_set),
            query_count_is_exact=exceeded_at is None, first_exceeding_family=exceeded_at,
            query_limit=limit, query_budget_pass=exceeded_at is None,
            object_limit=200000, object_budget_pass=len(profiles)<=200000,
            no_candidate_truncation=True, all_cells_stored=True, features_ready=False,
            target_reads=0, STEP_reads=0, optimizer_updates=0,
            boundary_evidence=boundary_meta, source_input_sha256=case['input_sha256'],
            domain_sha256=db['json_sha256'], point_feature_sha256=fb['feature_sha256'],
            files={p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='progress.json'},
            seconds=time.perf_counter()-start, utc=utc())
        write(dest/'receipt.json', receipt)
        print('ALL_PARENT_CELLS_DONE', key, receipt['cells'], 'QUERY_BUDGET_PASS', receipt['query_budget_pass'], flush=True)
    except BaseException:
        write(dest/'failure.json', dict(error=traceback.format_exc(), utc=utc(), optimizer_updates=0))
        raise


if __name__ == '__main__':
    main()
