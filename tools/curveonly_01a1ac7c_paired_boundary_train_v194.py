"""TRAIN-only paired boundary supervision with case/group-balanced losses."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_sequence_train_v172 as configure
from weld_pipeline import curveonly_01a1ac7c_paired_boundary_v193 as m

SETTINGS = dict(seed=20261106, epochs=128, learning_rate=.003, final_learning_rate=.0003,
    weight_decay=.0001, gradient_norm_limit=5., CPU_threads=2, GPU_memory_fraction=.35,
    batch_size=m.base.BATCH_SIZE, channels=m.base.CHANNELS, kernel=m.base.KERNEL,
    dilations=list(m.base.DILATIONS), observations=111, native_grid_max_mm=2.,
    foreground_loss_weight=1., boundary_loss_weight=2., offset_loss_weight=1.,
    local_order_loss_weight=.5, full_cycle_loss_weight=1., offset_smooth_beta=.1,
    normalizer='TRAIN_complement_midrange_half_range_no_clipping',
    supervision='paired_grid_boundaries_independent_offsets_same_cell_order_full_cycle')


def class_weights(rows, canonical, head, positive):
    groups = [np.where(q['targets'][head] > 0, 0, 1 if q['candidate'] in canonical else 2)
              for q in rows]
    counts = np.array([sum(int((g == k).sum()) for g in groups) for k in range(3)])
    present = int((counts > 0).sum())
    assert present > 0
    for row, group, pos in zip(rows, groups, positive):
        weight = np.zeros(len(group), np.float32)
        take = group == 0
        weight[take] = pos[take]/present
        for k in (1, 2):
            take = group == k
            if take.any():
                weight[take] = 1/(present*counts[k])
        row['class_weight'][head] = weight
    assert abs(sum(float(q['class_weight'][head].sum()) for q in rows)-1) < 1e-5
    return counts.tolist()


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


def loss_terms(point, full, batch):
    classification = (F.binary_cross_entropy_with_logits(point[:, :3], batch['targets'][:, :3],
                         reduction='none')*batch['class_weight']).sum((0, 2))
    offset = .5*torch.tanh(point[:, 3:5])
    reg = (F.smooth_l1_loss(offset, batch['targets'][:, 3:5], reduction='none',
                          beta=SETTINGS['offset_smooth_beta'])*batch['offset_weight']).sum()
    order = (F.binary_cross_entropy_with_logits(point[:, 5], batch['targets'][:, 5],
               reduction='none')*batch['order_weight']).sum()
    closed = (F.binary_cross_entropy_with_logits(full, batch['full_target'],
                reduction='none')*batch['full_weight']).sum()
    return classification, reg, order, closed


def objective(terms, masses):
    classification, reg, order, closed = terms
    return (SETTINGS['foreground_loss_weight']*classification[0]/masses['classification']+
            SETTINGS['boundary_loss_weight']*classification[1:].mean()/masses['classification']+
            SETTINGS['offset_loss_weight']*reg/max(1, masses['offset'])+
            SETTINGS['local_order_loss_weight']*order/max(1, masses['order'])+
            SETTINGS['full_cycle_loss_weight']*closed/max(1, masses['full']))


def assembly_control(cases, destination):
    configure.configure(SETTINGS)
    rows, mean, scale, masses, details = assemble(cases)
    ids = max(m.base.batches(rows), key=lambda ids: sum(float(rows[i]['offset_weight'].sum()) for i in ids))
    batch = m.pack([rows[i] for i in ids], mean, scale, 'cuda')
    network = m.BoundaryNetwork().cuda()
    a, b, af, bf = network.both_directions(batch)
    loss = .5*(objective(loss_terms(a, af, batch), masses)+objective(loss_terms(b, bf, batch), masses))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 5., error_if_nonfinite=True)
    assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
    e.write(Path(destination)/'ASSEMBLY_CONTROL.json', dict(passed=True, masses=masses,
        TRAIN_inputs=len(cases), details=details, finite_real_GPU_loss_and_backward=True,
        no_optimizer_step_or_saved_random_weights=True, parameter_count=sum(p.numel() for p in network.parameters())))
    torch.cuda.empty_cache()


def fit(cases, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    configure.configure(SETTINGS)
    rows, mean, scale, masses, details = assemble(cases)
    network = m.BoundaryNetwork().cuda().train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=SETTINGS['learning_rate'],
                                  weight_decay=SETTINGS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SETTINGS['epochs'],
                                                          eta_min=SETTINGS['final_learning_rate'])
    latest = destination/'LATEST_CHECKPOINT.pt'
    losses, first_epoch = [], 0
    if latest.exists():
        saved = torch.load(latest, map_location='cuda', weights_only=False)
        assert saved['settings'] == SETTINGS and saved['source_ids'] == sorted(q['source_id'] for q in cases)
        assert np.array_equal(mean, saved['mean']) and np.array_equal(scale, saved['scale'])
        network.load_state_dict(saved['network'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        losses, first_epoch = saved['losses'], saved['epoch']
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, SETTINGS['epochs']):
        begin = time.perf_counter()
        groups = m.base.batches(rows, np.random.default_rng(SETTINGS['seed']+epoch))
        totals = np.zeros(7)
        for ids in groups:
            selected = [rows[i] for i in ids]
            batch = m.pack(selected, mean, scale, 'cuda')
            optimizer.zero_grad(set_to_none=True)
            a, b, af, bf = network.both_directions(batch)
            value = 0.
            for point, full in ((a, af), (b, bf)):
                terms = loss_terms(point, full, batch)
                value += .5*len(groups)*objective(terms, masses)
                with torch.no_grad():
                    step = torch.as_tensor([q['grid_step_mm'] for q in selected],
                                            dtype=torch.float32, device='cuda')[:, None, None]
                    error = ((.5*torch.tanh(point[:, 3:5])-batch['targets'][:, 3:5]).abs()*step*
                             batch['offset_weight']).sum()
                    cls, reg, order, cycle = terms
                    totals += .5*np.r_[cls.detach().cpu().numpy(), float(reg.detach()),
                                       float(order.detach()), float(cycle.detach()), float(error)]
            assert torch.isfinite(value).item()
            value.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), SETTINGS['gradient_norm_limit'], error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        record = dict(epoch=epoch+1,
            foreground_loss=float(totals[0]/masses['classification']),
            start_loss=float(totals[1]/masses['classification']),
            end_loss=float(totals[2]/masses['classification']),
            offset_loss=float(totals[3]/max(1, masses['offset'])),
            order_loss=float(totals[4]/max(1, masses['order'])),
            full_cycle_loss=float(totals[5]/max(1, masses['full'])),
            TRAIN_local_offset_mean_mm=float(totals[6]/max(1, masses['offset'])),
            seconds=time.perf_counter()-begin, learning_rate=float(scheduler.get_last_lr()[0]))
        losses.append(record)
        e.write(destination/'TRAIN_PROGRESS.json', dict(settings=SETTINGS, epochs=losses,
            local_offset_diagnostic_not_complete_geometry_accuracy=True))
        print('PAIRED_BOUNDARY_TRAIN_EPOCH', record, flush=True)
        if (epoch+1) % 16 == 0 or epoch+1 == SETTINGS['epochs']:
            path = destination/f'epoch_{epoch+1:04d}.pt'
            torch.save(dict(network=network.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch+1, settings=SETTINGS,
                source_ids=sorted(q['source_id'] for q in cases), mean=mean, scale=scale,
                losses=losses), path)
            link = destination/'LATEST_CHECKPOINT.link'
            if link.is_symlink():
                link.unlink()
            link.symlink_to(path.name)
            os.replace(link, latest)
    peak = int(torch.cuda.max_memory_allocated())
    network = network.cpu().eval()
    torch.save(dict(network=network.state_dict(), mean=mean, scale=scale, settings=SETTINGS),
               destination/'weights.pt')
    saved = torch.load(destination/'weights.pt', map_location='cpu', weights_only=False)
    restored = m.BoundaryNetwork()
    restored.load_state_dict(saved['network'])
    before = m.predict_fields(network, rows, mean, scale)
    after = m.predict_fields(restored, rows, saved['mean'], saved['scale'])
    assert all(np.array_equal(a, b) for a, b in zip(before[0], after[0]))
    assert np.array_equal(before[1], after[1])
    record = dict(weights=e.bind(destination/'weights.pt'), TRAIN_inputs=len(cases), cases=details,
        settings=SETTINGS, masses=masses, parameter_count=sum(q.numel() for q in network.parameters()),
        GPU_peak_allocated_bytes=peak, epochs=losses, elapsed_seconds=time.perf_counter()-start,
        full_fit_population_CPU_reload_fields_bit_exact=True,
        no_held_or_VALfit_or_epoch_choice=True, no_old_fitted_models_used=True)
    e.write(destination/'FIT.json', record)
    torch.cuda.empty_cache()
    return network, mean, scale, record
