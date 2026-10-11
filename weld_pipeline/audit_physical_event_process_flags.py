"""Describe missing physical locations by original process flags; never infer with labels."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.labels import sha256

AUDIT = Path('weld_pipeline/reports/physical_event_candidates_161_20261002')
OUT = Path('weld_pipeline/reports/physical_event_process_flags_162_20261002')


def main():
    if OUT.exists():
        raise ValueError('Preserve original process-flag diagnostic')
    config = json.loads((AUDIT/'config.json').read_text('utf8'))
    done = json.loads((AUDIT/'complete.json').read_text('utf8'))
    rows = []; sources = {}
    for name in config['train']:
        path = AUDIT/(name+'.json')
        if sha256(path) != done['piece_receipts'][name]:
            raise ValueError('Full physical coverage source changed')
        audited = json.loads(path.read_text('utf8'))
        relation = json.loads((Path('weld_pipeline/data/joint_relation_v1')/name/'manifest.json').read_text('utf8'))
        mf = Path(relation['source_cache_root'])/'manifest.json'
        if sha256(mf) != audited['source_manifest_sha256']:
            raise ValueError('Original source labels changed')
        manifest = json.loads(mf.read_text('utf8'))
        welds = {w['id']:w for w in manifest['truth']}
        sources[name] = dict(audit_sha256=sha256(path), source_manifest_sha256=sha256(mf))
        for child in audited['children']:
            weld = welds[child['parent_id']]; index = child['segment']; meta = weld.get('metadata', {})
            for end in (0, 1):
                at_start = index == 0 and end == 0
                at_end = index == len(weld['segments'])-1 and end == 1
                role = 'outer_start' if at_start else 'outer_end' if at_end else 'internal'
                flag = meta.get('WorkRangeFrontSplit') if at_start else meta.get('WorkRangeBackSplit') if at_end else None
                point = np.asarray(weld['segments'][index]['points'], float)[0 if end == 0 else -1]
                rows.append(dict(piece=name, parent_id=weld['id'], segment=index, end=end, role=role,
                    point_mm=point.tolist(), explicit_endpoint_workrange_split=flag,
                    parent_workrange_any=meta.get('IsWorkRangeSplit (Any)'),
                    parent_dual_robot_any=meta.get('IsSplitedForDualRobot (Any)'),
                    finite_error_mm=child['finite']['endpoints'][end]['error_mm'],
                    unlimited_error_mm=child['unlimited']['endpoints'][end]['error_mm']))
    if len(rows) != 2442:
        raise ValueError('All original child endpoints must remain in the diagnostic')
    groups = defaultdict(list)
    for row in rows:
        for key in ('role', 'explicit_endpoint_workrange_split', 'parent_workrange_any', 'parent_dual_robot_any'):
            groups[key+'='+str(row[key])].append(row)
    statistics = {}
    for key, records in groups.items():
        errors = np.array([r['finite_error_mm'] for r in records])
        statistics[key] = dict(total=len(records), covered_3mm=int((errors <= 3.).sum()),
            missing_3mm=int((errors > 3.).sum()), median_error_mm=float(np.median(errors)),
            p90_error_mm=float(np.quantile(errors, .9)), source_count=len({r['piece'] for r in records}))
    clusters = Counter(round(row['point_mm'][0], 1) for row in rows if row['finite_error_mm'] > 3.)
    repeated = []
    for x, count in clusters.most_common(12):
        records = [r for r in rows if r['finite_error_mm'] > 3. and round(r['point_mm'][0], 1) == x]
        repeated.append(dict(raw_X_mm=x, occurrences=count, pieces=sorted({r['piece'] for r in records}),
            parent_workrange_true=sum(r['parent_workrange_any'] is True for r in records),
            explicit_endpoint_workrange_true=sum(r['explicit_endpoint_workrange_split'] is True for r in records)))
    OUT.mkdir(parents=True)
    (OUT/'endpoint_rows.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
    receipt = dict(audit_complete_sha256=sha256(AUDIT/'complete.json'), sources=sources,
        original_child_endpoint_count=len(rows), groups=statistics,
        repeated_raw_coordinate_patterns_diagnostic_only=repeated,
        no_coordinate_lookup_rule_or_new_runtime_configuration_created=True,
        original_flags_do_not_prove_endpoint_cause_or_PC_observability=True,
        unknown_process_rules_not_frozen=True, no_labels_changed=True,
        observed_or_new_held_read=False, diagnostic_only=True, full_contract_pass=False,
        source_sha256=sha256(__file__), rows_sha256=sha256(OUT/'endpoint_rows.json'))
    (OUT/'complete.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    print('PHYSICAL_EVENT_PROCESS_FLAGS', statistics, flush=True)
    print('REPEATED_RAW_X_DIAGNOSTIC_ONLY', repeated, flush=True)


if __name__ == '__main__':
    main()
