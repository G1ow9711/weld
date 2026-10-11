"""Rebuild all 97 point/255 node inputs directly from PCD, with no reference reads."""
import builtins
import io
import json
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.fresh_point_context import point_structure
from weld_pipeline.joint_relation_heatmaps import make_graph, LOCAL_CHANNELS
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.workpiece_relative_context import WorkpieceRelativeContext

BASE = Path('weld_pipeline/data/joint_relation_v1')
DATA = Path('weld_pipeline/data/joint_workpiece_context_pilot154')
OUT = Path('weld_pipeline/reports/joint_workpiece_context_raw_154_20261001')


def main():
    if OUT.exists():
        raise ValueError('Preserve raw neural input proof')
    cfg = json.loads((DATA/'config.json').read_text('utf8'))
    for source, digest in cfg['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Frozen joint context source changed')
    OUT.mkdir(parents=True); results = {}
    for name in ('207-FR79M', '203-TT1A'):
        manifest = json.loads((DATA/name/'manifest.json').read_text('utf8'))
        pcd = Path(manifest['input']).resolve(); before = sha256(pcd)
        original_open = (builtins.open, io.open, Path.open)
        forbidden = [Path(p).resolve() for p in ('DataSet', 'weld_pipeline/data', 'weld_pipeline/reports')]
        reads = []; controls = []
        def guard(opener):
            def checked(path, mode='r', *args, **kwargs):
                if isinstance(path, (str, bytes, Path)) and ('r' in str(mode) or '+' in str(mode)):
                    file = Path(path).resolve()
                    if file == pcd:
                        reads.append(str(file))
                    elif (any(file.is_relative_to(root) for root in forbidden)
                          or file.suffix.lower() in ('.step', '.stp', '.xlsx', '.xls')
                          or file.name.endswith('_welds.json')):
                        raise RuntimeError('Forbidden joint runtime reference/cache: '+str(file))
                return opener(path, mode, *args, **kwargs)
            return checked
        started = time.perf_counter()
        builtins.open, io.open, Path.open = tuple(guard(opener) for opener in original_open)
        try:
            for opener in (builtins.open, io.open, Path.open):
                try:
                    opener(Path('weld_pipeline/data/joint_context_control.step'), 'r')
                except RuntimeError:
                    controls.append(True)
                else:
                    raise ValueError('Joint reference/cache read guard failed')
            xyz = read_xyz_pcd(pcd)*1000.; sequences, _ = make_mixed_sequences(xyz)
            structural, process = point_structure(xyz, sequences)
            graph = make_graph(xyz, sequences, structural, process, FiniteContactPlates(xyz))
            context = WorkpieceRelativeContext(xyz)
            local_blocks = []; point_blocks = []; nodes = []
            for index in graph['sequence_indices']:
                sequence, _, _ = canonical_frame(sequences[int(index)])
                local_blocks.append(sequence['features'][:, LOCAL_CHANNELS])
                point_blocks.append(context.describe_sequence(sequence, np.arange(len(sequence['xyz']))))
                mid = (sequence['xyz'][0].astype(float)+sequence['xyz'][-1].astype(float))/2.
                ns = sequence['features'][len(sequence['xyz'])//2, 60:66].reshape(2, 3)
                nodes.append(context.describe(mid, ns, sequence['direction']))
            fresh_local = np.concatenate(local_blocks); fresh_point = np.concatenate(point_blocks)
            fresh_node = np.asarray(nodes, np.float32)
        finally:
            builtins.open, io.open, Path.open = original_open
        base_local = np.load(BASE/name/'features.npy', allow_pickle=False)
        base_context = np.load(DATA/name/'point_context.npy', allow_pickle=False)
        expected_node = np.load(DATA/name/'node_context.npy', allow_pickle=False)
        with np.load(BASE/name/'graph.npz', allow_pickle=False) as expected_graph:
            graph_equal = all(np.array_equal(graph[k], expected_graph[k]) for k in
                ('nodes', 'edge_index', 'edge_type', 'edge_features', 'sequence_indices'))
        local_equal = np.array_equal(fresh_local, base_local)
        point_equal = np.array_equal(fresh_point, base_context)
        node_equal = np.array_equal(fresh_node, expected_node)
        passed = bool(graph_equal and local_equal and point_equal and node_equal
            and reads and len(controls) == 3 and sha256(pcd) == before)
        result = dict(piece=name, rows=len(fresh_local), nodes=len(fresh_node), graph_exact=graph_equal,
            local_point_exact=local_equal, added_point_context_exact=point_equal, added_node_context_exact=node_equal,
            fresh_reads=len(reads), negative_controls=len(controls), input_sha256=before,
            elapsed_seconds=time.perf_counter()-started, all_checks_passed=passed,
            runtime_labels_used=False, runtime_STEP_used=False, runtime_preprocessed_cache_used=False,
            full_contract_pass=False)
        (OUT/(name+'.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
        print('JOINT_WORKPIECE_RAW_INPUT', name, passed, 'PARTS', graph_equal, local_equal, point_equal, node_equal, flush=True)
        results[name] = result
        if not passed:
            raise ValueError('Fresh 97/255 inputs differ from training: '+name)
    (OUT/'complete.json').write_text(json.dumps(dict(pieces=results, all_checks_passed=True,
        source_sha256={p:sha256(p) for p in [__file__, 'weld_pipeline/workpiece_relative_context.py',
            'weld_pipeline/fresh_point_context.py', 'weld_pipeline/joint_relation_heatmaps.py',
            'weld_pipeline/mixed_sequences.py', 'weld_pipeline/canonical_horizontal_patch_heatmaps.py']},
        observed_check_read=False, full_contract_pass=False), indent=2, ensure_ascii=False), encoding='utf8')


if __name__ == '__main__':
    main()
