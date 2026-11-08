"""Same paired-boundary optimizer/loss; repair only shared-cell identity targets."""
import types
from tools import curveonly_01a1ac7c_paired_boundary_train_v194 as previous
from weld_pipeline import curveonly_01a1ac7c_instance_order_v197 as m

SETTINGS = dict(previous.SETTINGS, same_cell_order_supervision='COMPLETE_INSTANCE_IDENTITY')
configure = previous.configure
np, e, class_weights = previous.np, previous.e, previous.class_weights


def assemble(cases):
    all_rows, details = [], []
    masses = dict(classification=0., offset=0., order=0., full=0.)
    for case in sorted(cases, key=lambda q: q['source_id']):
        assert case['split'] == 'TRAIN'
        for key in ('features', 'geometry', 'local_targets'):
            assert e.sha(case[key]['path']) == case[key]['sha256']
        features, packet = np.load(case['features']['path']), e.read(case['geometry']['path'])
        rows = m.candidate_rows(features, packet)
        canonical = {q['candidate'] for q in case['chosen']}
        count = len(case['chosen'])
        nonfull = []
        prepared, occupancy_weights = [], []
        full_groups = []
        for row in rows:
            chosen = [q for q in case['chosen'] if q['candidate'] == row['candidate']]
            target = m.targets(row, chosen)
            n = len(row['grid'])
            row.update(targets=target['fields'].astype(np.float32),
                class_weight=np.zeros((3, n), np.float32), offset_weight=np.zeros((2, n), np.float32),
                order_weight=np.zeros(n, np.float32), full_target=target['full'], full_weight=0.,
                source_id=case['source_id'])
            prepared.append(target)
            nonfull.extend(q for q in target['codes'] if not q['full_cycle'])
            position = np.arange(n)*row['grid_step_mm']
            assigned = np.full(n, -1)
            for i, (choice, code) in enumerate(zip(chosen, target['codes'])):
                take = np.zeros(n, bool)
                if code['full_cycle']:
                    take[:] = True
                else:
                    for turn in ((-1, 0, 1) if row['closed'] else (0,)):
                        s = position+turn*row['length_mm']
                        take |= (s >= choice['start_mm']) & (s <= choice['end_mm'])
                if not take.any():
                    take[[code['start_index'], code['end_index']]] = True
                assert not np.any(take & (assigned >= 0)), 'Overlapping canonical grid instance targets'
                assigned[take] = i
            pos = np.zeros(n, float)
            for i in range(len(chosen)):
                take = assigned == i
                assert take.any()
                pos[take] = 1/(count*int(take.sum()))
            occupancy_weights.append(pos)
            full_groups.append(0 if target['full'] else 1 if row['candidate'] in canonical else 2)
        nonfull_count = len(nonfull)
        groups = [class_weights(rows, canonical, 0, occupancy_weights)]
        for head in (1, 2):
            positive = [q['targets'][head].astype(float)/max(1, nonfull_count) for q in rows]
            groups.append(class_weights(rows, canonical, head, positive))
        order_count = sum(int(q['order_known'].sum()) for q in prepared)
        for row, target in zip(rows, prepared):
            row['offset_weight'] = row['targets'][1:3]/max(1, 2*nonfull_count)
            if order_count:
                row['order_weight'][target['order_known']] = 1/order_count
        closed_counts = np.array([sum(row['closed'] and group == k
                                     for row, group in zip(rows, full_groups)) for k in range(3)])
        present = int((closed_counts > 0).sum())
        if present:
            for row, group in zip(rows, full_groups):
                if row['closed']:
                    row['full_weight'] = 1/(present*closed_counts[group])
            assert abs(sum(q['full_weight'] for q in rows)-1) < 1e-6
        masses['classification'] += 1
        masses['offset'] += nonfull_count > 0
        masses['order'] += order_count > 0
        masses['full'] += present > 0
        all_rows.extend(rows)
        details.append(dict(source_id=case['source_id'], candidates=len(rows),
            complete_intervals=count, nonfull_intervals=nonfull_count,
            full_cycles=sum(int(q['full']) for q in prepared),
            shared_start_end_cells=order_count, class_group_counts=groups,
            closed_curve_group_counts=closed_counts.tolist(),
            old_targets_retained_as_audit_only=case['local_targets'],
            new_grid_instance_supervision_not_asserted_identical_to_old_station_loss=True))
    mean, scale = m.codec.previous.fit_range_normalizer(all_rows)
    maximum = max(float(np.abs((q['grid']-mean)/scale).max()) for q in all_rows)
    assert maximum <= 1.00001
    return all_rows, mean, scale, masses, details


namespace = dict(previous.__dict__, m=m, SETTINGS=SETTINGS, assemble=assemble)
assembly_control = types.FunctionType(previous.assembly_control.__code__, namespace)
fit = types.FunctionType(previous.fit.__code__, namespace)
