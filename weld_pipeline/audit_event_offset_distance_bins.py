"""FIT-only localization error versus true event distance, with unchanged frozen inputs."""
import json
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.joint_workpiece_event_votes import JointWorkpieceEventVotes
from weld_pipeline.train_joint_workpiece_event_capacity import attach_context_and_targets, TARGETS
from weld_pipeline.train_joint_relation_crossfit import DATA, load_data

RUN = Path('weld_pipeline/runs/joint_workpiece_event_capacity_seed154')
OUT = Path('weld_pipeline/reports/event_offset_distance_bins_156_20261001')
NAMES = ('161-LB18A', '203-TT1A', '207-FR80A')


@torch.no_grad()
def raw_predictions(model, item):
    model.eval(); context = model.encode_graph(*item['torch_graph'])
    raw = np.full((len(item['features']), 7), np.nan, np.float32)
    blocks = []
    for entry in item['horizontal']:
        length = entry['stop']-entry['start']
        for a in range(0, length, 1024):
            blocks.append((entry, a, min(a+1024, length)))
    for at in range(0, len(blocks), 8):
        batch = blocks[at:at+8]; size = max(b-a+128 for _, a, b in batch)
        x = np.zeros((len(batch), 97, size), np.float32); nodes = []
        for bi, (entry, a, b) in enumerate(batch):
            lo = a-64; left, right = max(0, lo), min(entry['stop']-entry['start'], b+64)
            sl = slice(entry['start']+left, entry['start']+right)
            x[bi, :75, left-lo:right-lo] = item['features'][sl].T
            x[bi, 75:, left-lo:right-lo] = item['point_context'][sl].T
            nodes.append(entry['node'])
        pred = model.point_forward(torch.from_numpy(x).cuda(), context[torch.tensor(nodes, device='cuda')]).cpu().numpy()
        for bi, (entry, a, b) in enumerate(batch):
            raw[entry['start']+a:entry['start']+b] = pred[bi, :, 64:64+b-a].T
    return raw


def main():
    if OUT.exists():
        raise ValueError('Preserve distance-bin diagnostic')
    protocol = json.loads((RUN/'config.json').read_text('utf8'))
    receipt = json.loads((RUN/'complete.json').read_text('utf8'))
    checkpoint = RUN/'final.pt'; frozen = sha256(checkpoint)
    if frozen != receipt['checkpoint_sha256'] or any(n not in protocol['train'] for n in NAMES):
        raise ValueError('Only the frozen train-source model may be analyzed')
    torch.set_num_threads(2)
    cfg = json.loads((DATA/'config.json').read_text('utf8'))
    data = load_data(dict(cfg, pieces=protocol['train']))
    attach_context_and_targets(data, protocol['train'])
    state = torch.load(checkpoint, map_location='cpu')
    model = JointWorkpieceEventVotes().cuda(); model.load_state_dict(state['model'])
    OUT.mkdir(parents=True); results = {}
    for name in NAMES:
        item = data[name]; raw = raw_predictions(model, item)
        target, known = item['vote_targets'], item['vote_known']; heads = {}
        for c, label in enumerate(('start', 'end', 'cut')):
            valid = (known[:, c+4] > 0)&np.isfinite(raw[:, c+4])
            distance = np.abs(target[valid, c+4])*64.
            error = np.abs(np.clip(raw[valid, c+4], -1., 1.)-target[valid, c+4])*64.
            support = 1./(1.+np.exp(-np.clip(raw[valid, c+1], -60., 60.)))
            bins = []
            for lo, hi in ((0., 8.), (8., 16.), (16., 32.), (32., 64.00001)):
                mask = (distance >= lo)&(distance < hi)
                e = error[mask]
                bins.append(dict(distance_min_mm=lo, distance_max_mm=min(hi, 64.), points=int(mask.sum()),
                    mean_error_mm=float(e.mean()) if len(e) else None,
                    p95_error_mm=float(np.quantile(e, .95)) if len(e) else None,
                    within_3mm=float((e<=3.).mean()) if len(e) else None,
                    support_recall=float((support[mask]>.5).mean()) if len(e) else None))
            heads[label] = bins
        results[name] = heads
        print('EVENT_OFFSET_DISTANCE_BINS', name, json.dumps(heads), flush=True)
    if sha256(checkpoint) != frozen:
        raise ValueError('Frozen precision model changed')
    (OUT/'complete.json').write_text(json.dumps(dict(pieces=results, checkpoint_sha256=frozen,
        target_manifest_sha256=sha256(TARGETS/'manifest.json'), diagnostic_only=True,
        held_sources_read=False, observed_check_read=False, full_contract_pass=False,
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/joint_workpiece_event_votes.py',
                                     'weld_pipeline/train_joint_workpiece_event_capacity.py']}),
        indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
