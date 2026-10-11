"""TRAIN-only chart adapter for original-direction 645/648 hypothesis caches."""
from weld_pipeline.legacy_incremental_supervision_388 import read, digest
from weld_pipeline.legal_allocation_training_data_520 import ROOT
from weld_pipeline.reachable_attribute_training_data_648 import prepare_complete_source
from weld_pipeline.legacy_incremental_correspondence_414 import canonical_child_fields
from weld_pipeline.contract_field_acceptance_297 import FIELDS

VERSION = 'TRAIN_PHYSICAL_HYPOTHESES_CANONICAL_POSITIVE_CHART_V1'


def canonical_hypothesis(record, negative):
    """Same physical witnesses, reindexed once to match path features/targets."""
    fields = canonical_child_fields(dict(zip(FIELDS, record['fields'])), negative)
    return dict(record, fields=[fields[name] for name in FIELDS])


def align_source_hypotheses(index, source, metadata):
    assert 'hypothesis_chart_alignment' not in source, 'Do not apply chart reversal twice'
    entry = metadata['rows'][index]['mapping']['hybrid']
    path = ROOT / entry['path']; assert digest(path) == entry['sha256']
    mapping = read(path)
    by_id = {cid: parent['child_parameter_endpoints_mm'][j][1] < parent['child_parameter_endpoints_mm'][j][0]
             for parent in mapping['mapped_parents'] for j,cid in enumerate(parent['original_child_ids'])}
    assert source['whole_path_fields_and_features_use_one_positive_chart_correspondence']
    records = source['path_plate_hypotheses']
    assert len(records) == len(source['original_child_ids'])
    negative = [bool(by_id.get(cid, False)) for cid in source['original_child_ids']]
    source['path_plate_hypotheses'] = [canonical_hypothesis(row, reverse) for row,reverse in zip(records,negative)]
    source['hypothesis_chart_alignment'] = dict(version=VERSION, mapping_sha256=entry['sha256'],
        negative_rows=sum(negative), original_cache_sha256=source['physical_hypothesis_cache_sha256'],
        original_cache_unchanged=True, local_point_hypotheses_unchanged=True,
        fields_features_and_targets_share_positive_chart=True)
    return source


def prepare_canonical_complete_source(index, prior, metadata, cache_path):
    return align_source_hypotheses(index, prepare_complete_source(index,prior,metadata,cache_path), metadata)
