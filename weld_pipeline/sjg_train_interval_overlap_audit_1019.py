"""Check original TRAIN child overlaps before adding joint exclusivity."""
from itertools import combinations
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc


def overlap(a, b):
    return sum(max(0., min(x[1], y[1])-max(x[0], y[0])) for x in a for y in b)


def main():
    data = OUT/'run186';rows = []
    for case in read(data/'training_manifest.json')['cases']:
        key = case['source_key'];assert key not in ('64','24')
        path = data/('source_'+key+'_supervision.json');sup = read(path)
        pairs = {};shared = set();separated = set()
        for profile, events in sup['by_profile'].items():
            for a,b in combinations(events,2):
                ia=(a['parent'],a['child']);ib=(b['parent'],b['child'])
                if ia==ib:continue
                identity=tuple(sorted((ia,ib)))
                amount=overlap(a['intervals'],b['intervals'])
                if amount>1e-5:
                    if identity not in pairs or amount>pairs[identity]['maximum_overlap_mm']:
                        pairs[identity]=dict(children=identity,maximum_overlap_mm=amount,profile=int(profile),
                            same_parent=ia[0]==ib[0],kinds=[a['kind'],b['kind']])
                else:
                    distance=min(abs(x-y) for x in (a['lo'],a['hi']) for y in (b['lo'],b['hi']))
                    (shared if distance<=1e-5 else separated).add(identity)
        row=dict(source_key=key,original_G=case['original_G'],supervision_sha256=digest(path),
            distinct_child_overlaps=list(pairs.values()),overlap_above_6mm=sum(v['maximum_overlap_mm']>6 for v in pairs.values()),
            exact_shared_endpoint_pairs=len(shared),separated_pairs=len(separated))
        rows.append(row)
        print('TRAIN_INTERVAL_OVERLAP',key,'G',case['original_G'],'OVERLAPS',len(pairs),
              'ABOVE_6MM',row['overlap_above_6mm'],'SHARED',len(shared),'GAPPED',len(separated),flush=True)
    assert sum(row['original_G'] for row in rows)==363
    write(OUT/'run197/TRAIN_shared_carrier_overlap_audit.json',dict(rows=rows,original_training_G=363,
        geometric_compatibility_not_material_identity=True,heldout_not_read=True,optimizer_updates=0,
        source_sha256=digest(__file__),end_utc=utc()))


if __name__=='__main__':main()
