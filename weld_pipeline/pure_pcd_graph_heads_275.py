"""Pure observed-source input reader and typed graph neural heads.

No STEP, teacher geometry, source-name feature, or selected truth domain is read
by SourceFeatures. Graph targets are supplied separately to the loss consumer.
This component alone is not a runtime decoder or a passed training round.
"""
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'PURE_PCD_FULL_POOL_TYPED_GRAPH_HEADS_275_v1'
CATEGORIES = {'span_kind': 4, 'world_kind': 4, 'weld_type': 2,
              'closed': 2, 'direction': 2, 'position_kind': 3}
RELATION_FIELDS = ('parent', 'children', 'start_boundary', 'end_boundary', 'geometry_spans',
                   'internal_geometry_nodes', 'phase_boundary', 'authority_span',
                   'physical_ref', 'curve_endpoint', 'gap_left', 'gap_right', 'gap_span')


def digest(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class SourceFeatures:
    """All candidate anchors and domains come from the complete 271 source pool."""
    def __init__(self, root, entry, *, verify_artifact_hashes=True):
        self.root = Path(root)
        directory = self.root / entry['feature_directory']
        producer_path = directory / 'receipt.json'
        review_path = directory / 'independent_complete_review_attempt1/receipt.json'
        assert digest(producer_path) == entry['producer_receipt_sha256']
        assert digest(review_path) == entry['independent_receipt_sha256']
        producer, review = read(producer_path), read(review_path)
        assert review['status'] == 'PASS_INDEPENDENT_COMPLETE_SOURCE_CANONICAL_ANCHOR_DOMAIN_CONTEXT_INPUTS'
        assert review['producer_receipt_sha256'] == entry['producer_receipt_sha256']
        assert producer['source_index'] == review['source_index'] == entry['source_index']
        assert producer['original_pcd_sha256'] == entry['original_pcd_sha256'] == review['original_pcd_sha256']
        assert producer['label_reads'] == producer['STEP_reads'] == 0
        assert producer['teacher_geometry_or_selected_domains_read'] == 0
        assert not producer['feature_schema']['computational_plane_axes_are_material_normals']
        assert producer['feature_schema']['material_channels'] == []
        self.arrays = {}
        for name, artifact in producer['artifacts'].items():
            path = self.root / artifact['path']
            if verify_artifact_hashes:
                assert digest(path) == artifact['sha256'], name
            if name.endswith('.npy'):
                self.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        self.carrier_ids = read(self.root / producer['artifacts']['carrier_ids.json']['path'])
        self.anchor_count = producer['counts']['canonical_anchors']
        assert self.arrays['anchor_features.npy'].shape == (self.anchor_count, 10)
        assert self.arrays['surface_domain_features.npy'].shape[1] == 21
        assert self.arrays['carrier_domain_features.npy'].shape[1] == 29
        assert self.arrays['whole_source_context.npy'].shape == (19,)
        self.surface_ids = np.asarray(self.arrays['surface_face_ids.npy'], dtype=np.int64)
        self.domain_records = [{'type': 'LEGACY_SURFACE', 'face_id': int(v)} for v in self.surface_ids]
        # Carrier IDs are observed metadata used for pointer interpretation only.
        self.domain_records += [{'type': 'RAW_CARRIER', 'carrier_id': v} for v in self.carrier_ids]
        assert len(self.carrier_ids) == len(self.arrays['carrier_domain_features.npy'])
        self.source_index = entry['source_index']
        self.initial_input_contract = dict(label_reads=0, STEP_reads=0,
             teacher_selected_domain_reads=0, material_unknowns_promoted_to_known_zero=False,
             complete_anchor_candidate_count=self.anchor_count,
             domains=len(self.domain_records), world_geometry_remains_float64=True)

    def batch(self, anchor_rows, device='cpu'):
        rows = np.asarray(anchor_rows, dtype=np.int64)
        if rows.ndim != 1 or (rows < 0).any() or (rows >= self.anchor_count).any():
            raise ValueError('Batch must index the unchanged full canonical anchor pool')
        names = {'anchors': 'anchor_features.npy', 'source': 'whole_source_context.npy',
                 'surfaces': 'surface_domain_features.npy', 'carriers': 'carrier_domain_features.npy'}
        result = {}
        for key, name in names.items():
            array = self.arrays[name][rows] if key == 'anchors' else self.arrays[name]
            result[key] = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)
        return result

    def domain_index(self, value):
        key = json.dumps(value, sort_keys=True, separators=(',', ':'))
        matches = [i for i, row in enumerate(self.domain_records)
                   if json.dumps(row, sort_keys=True, separators=(',', ':')) == key]
        if len(matches) != 1:
            raise ValueError('Target domain is absent or ambiguous in the complete observed source')
        return matches[0]


class SourceGraphHeads(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        def encoder(channels):
            return nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.anchor_encoder = encoder(10)
        self.context_encoder = encoder(19)
        self.surface_encoder = encoder(21)
        self.carrier_encoder = encoder(29)
        self.domain_type = nn.Embedding(2, hidden)
        self.context_attention = nn.Linear(hidden, hidden, bias=False)
        self.fusion = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.presence = nn.Linear(hidden, 6)
        self.numeric = nn.Linear(hidden, 6 * 12)
        self.category = nn.ModuleDict({name: nn.Linear(hidden, 6 * count) for name, count in CATEGORIES.items()})
        self.domain_query = nn.Linear(hidden, 6 * hidden)
        self.role_embedding = nn.Embedding(6, hidden)
        self.relation_embedding = nn.Embedding(len(RELATION_FIELDS), hidden)
        self.relation_score = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, inputs, *, observed_domain_mask=None):
        a = self.anchor_encoder(inputs['anchors'])
        context = self.context_encoder(inputs['source']).expand(len(a), -1)
        surface = self.surface_encoder(inputs['surfaces']) + self.domain_type.weight[0]
        carrier = self.carrier_encoder(inputs['carriers']) + self.domain_type.weight[1]
        domains = torch.cat((surface, carrier), dim=0)
        if not len(domains):
            raise ValueError('An observed domain pool is required, never replace unknown material by zero')
        logits = self.context_attention(a) @ domains.T / math.sqrt(self.hidden)
        if observed_domain_mask is not None:
            if observed_domain_mask.shape != logits.shape or observed_domain_mask.dtype != torch.bool:
                raise ValueError('Only an explicit source-derived domain incidence mask is allowed')
            if (~observed_domain_mask.any(dim=1)).any():
                raise ValueError('Empty domain incidence requires an explicit unknown outcome')
            logits = logits.masked_fill(~observed_domain_mask, -torch.inf)
        attended = logits.softmax(dim=1) @ domains
        hidden = self.fusion(torch.cat((a, context, attended), dim=1))
        return {'presence_logits': self.presence(hidden), 'numeric': self.numeric(hidden).reshape(-1, 6, 12),
                'categories': {name: layer(hidden).reshape(-1, 6, CATEGORIES[name]) for name, layer in self.category.items()},
                'domain_queries': self.domain_query(hidden).reshape(-1, 6, self.hidden),
                'domain_embeddings': domains, 'anchor_embeddings': hidden}

    def relation_logits(self, left, right, left_roles, right_roles, kinds):
        left = left + self.role_embedding(left_roles)
        right = right + self.role_embedding(right_roles)
        query = self.relation_embedding(kinds)
        return self.relation_score(torch.cat((left, right, query), dim=-1)).squeeze(-1)


def relation_kind(path):
    key = path[0]
    if key == 'gaps':
        tail = path[-1]
        key = {'left_boundary': 'gap_left', 'right_boundary': 'gap_right',
               'geometry_spans': 'gap_span'}.get(tail, 'gap_span')
    if key not in RELATION_FIELDS:
        raise ValueError('Graph relationship has no predictor: ' + str(path))
    return RELATION_FIELDS.index(key)


def supervised_loss(output, occupancy, anchor_batch_rows, role_columns, categories,
                    numerical, numerical_known, domain_targets=None):
    """Unknown labels contribute zero; sparse geometry remains separate from input."""
    if occupancy.shape != output['presence_logits'].shape or not torch.all((occupancy >= -1) & (occupancy <= 1)):
        raise ValueError('Complete role-mask dimensions or label states differ')
    logits = output['presence_logits']
    pieces, metrics = [], {}
    for label in (1, 0):
        mask = occupancy == label
        if mask.any():
            pieces.append(F.binary_cross_entropy_with_logits(logits[mask], torch.full_like(logits[mask], float(label))))
    presence = sum(pieces, logits.sum() * 0)
    metrics['presence'] = presence
    selected_numeric = output['numeric'][anchor_batch_rows, role_columns]
    if selected_numeric.shape != numerical.shape or numerical_known.shape != numerical.shape:
        raise ValueError('Numerical supervision schema changed')
    metrics['numeric'] = (F.smooth_l1_loss(selected_numeric[numerical_known], numerical[numerical_known])
                          if numerical_known.any() else selected_numeric.sum() * 0)
    if categories.shape != (len(anchor_batch_rows), len(CATEGORIES)):
        raise ValueError('Category supervision schema changed')
    category_loss = logits.sum() * 0
    for column, (name, count) in enumerate(CATEGORIES.items()):
        target = categories[:, column]
        known = target >= 0
        if known.any():
            assert (target[known] < count).all()
            prediction = output['categories'][name][anchor_batch_rows[known], role_columns[known]]
            category_loss = category_loss + F.cross_entropy(prediction, target[known])
    metrics['categories'] = category_loss
    domain_loss = logits.sum() * 0
    if domain_targets is not None:
        known = domain_targets >= 0
        if known.any():
            queries = output['domain_queries'][anchor_batch_rows[known], role_columns[known]]
            prediction = queries @ output['domain_embeddings'].T / math.sqrt(queries.shape[-1])
            domain_loss = F.cross_entropy(prediction, domain_targets[known])
    metrics['domain_pointer'] = domain_loss
    return sum(metrics.values()), metrics
