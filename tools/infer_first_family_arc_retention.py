"""Freeze pointer outputs on every input carrier, without reference-selected carriers."""
from pathlib import Path
from collections import Counter
import time
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_pointer import ArcBoundaryPointer
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

BASE = Path(__file__).resolve().parents[1] / 'server_rebuild/expansion_v1'


def main(model_name='curved_group_comparison_grouped_v1', output_name='arc_pointer_first_family_retention_v1'):
    started = time.perf_counter()
    torch.set_num_threads(2)
    key = 'new_ac2008a03607'
    source = BASE / model_name
    receipt = read(source / 'receipt.json')
    assert digest(source / 'model.pt') == receipt['model_sha256']
    protocol = read(source / 'protocol.json')
    checkpoint = torch.load(source / 'model.pt', map_location='cpu', weights_only=False)
    assert checkpoint['protocol_sha256'] == digest(source / 'protocol.json')
    features = BASE / 'training_arc_event_features_v1' / key
    events = BASE / 'training_arc_vertex_events_v1' / key
    fr = read(features / 'receipt.json')
    assert digest(features / 'protocol.json') == fr['protocol_sha256']
    assert read(features / 'protocol.json')['event_receipt_sha256'] == digest(events / 'receipt.json')
    assert digest('weld_pipeline/server_arc_boundary_pointer.py') == protocol['base_module_sha256']
    out = BASE / output_name
    out.mkdir(exist_ok=False)
    write(out / 'protocol.json', dict(key=key, model_sha256=receipt['model_sha256'], feature_receipt_sha256=digest(features / 'receipt.json'), event_receipt_sha256=digest(events / 'receipt.json'), candidate_selection='All frozen input carriers, no reference filter', decoding='Original unmodified ordered pointer and STOP; no count cap, geometry correction, deduplication or label-supplied type', scope='Retention on first TRAIN family after second-family continuation. First-family plane graph events have zero source height, full finite graph support, and actual vertex flag. All input carriers, no target reads or feature refit. Extra7 semantics aligned to second-family descriptors; no material/contact proof.', GT_reads=0, DEV_reads=0))
    model = ArcBoundaryPointer(input_dim=83).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    mean, std = checkpoint['mean'], checkpoint['std']
    records = []
    counts = Counter()
    for record in fr['faces']:
        path = features / record['file']
        ep = events / f"face_{record['face']}.npz"
        assert digest(path) == record['sha256']
        assert digest(ep) == record['event_file_sha256']
        x = np.load(path, mmap_mode='r')
        with np.load(ep) as z:
            offsets, positions, vertex_ids = z['offsets'], z['positions_mm'], z['vertex_indices']
        assert offsets[-1] == len(x) == len(positions)
        rows = []
        for index, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            assert b > a
            local_q=positions[a:b];extra=np.column_stack((np.zeros((b-a,2)),vertex_ids[a:b]>=0,np.ones(b-a),np.log1p(local_q),np.log1p(local_q[-1]-local_q),np.zeros(b-a)))
            raw=np.column_stack((x[a:b],extra));values = torch.as_tensor(np.clip((raw - mean) / std, -20, 20), dtype=torch.float32, device='cuda')
            q = torch.as_tensor(positions[a:b], dtype=torch.float64, device='cuda')
            selected = model.decode(values, q)
            assert all(positions[a+j] > positions[a+i] for i, j in zip(selected, selected[1:]))
            rows.append(dict(candidate=index, selected_local_event_indices=selected))
            counts[len(selected)] += 1
        target = out / f"face_{record['face']}.json"
        write(target, dict(face=record['face'], rows=rows))
        records.append(dict(face=record['face'], candidates=len(rows), file=target.name, sha256=digest(target)))
        if len(records) % 25 == 0:
            print('faces', len(records), 'carriers', sum(counts.values()), 'boundary_counts', dict(counts), flush=True)
    write(out / 'receipt.json', dict(status='ALL_INPUT_CARRIER_POINTER_OUTPUTS_FROZEN_NOT_PROMOTED', faces=records, protocol_sha256=digest(out / 'protocol.json'), carriers=sum(counts.values()), boundary_count_histogram=dict(counts), predicted_children=sum(max(0,n-1)*v for n,v in counts.items()), invalid_one_boundary=counts[1], immediate_stop=counts[0], GT_reads=0, DEV_reads=0, seconds=time.perf_counter()-started))
    print('COMPLETE', dict(counts), flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model-name',default='curved_group_comparison_grouped_v1');p.add_argument('--output-name',default='arc_pointer_first_family_retention_v1');a=p.parse_args();main(a.model_name,a.output_name)
