"""Reuse input geometry while evaluating the original run446 relation prior.

The same ParentRelations.pair and symmetric 32-column features are called.
Only decoded input primitives, sampled points and metadata are cached, without
opening supervision or changing parent geometry. Each cache belongs to one
workpiece and is released before moving to the next one.
"""
import numpy as np

from .server_full_prediction_runtime import sampled_segment
from .sjg_additive_parent_precision_1254 import pair_features
from .sjg_parent_set_features_1160 import ParentRelations


class CachedRun446RelationPrior:
    def __init__(self, priors):
        self.priors = priors
        self.checkpoint_sha256 = priors.checkpoint_sha256
        self.clear()

    def clear(self):
        self.parents = None
        self.cache = {}
        self.parent_geometry_builds = 0
        self.relation_rows = 0

    def entry(self, index):
        index = int(index)
        if index not in self.cache:
            parent = self.parents[index]
            segments = []
            for child in parent['segments']:
                represented = sampled_segment(child)
                # The legacy relation method already converts these points to
                # float64 arrays. Keep that value once instead of Python lists.
                represented['points'] = np.asarray(represented['points'], np.float64)
                segments.append(represented)
            decoded = dict(parent, segments=segments)
            children = parent['segments']
            length = sum(c['length_mm'] for c in children)
            metadata = dict(face_pairs=[pair for c in children for pair in c['face_pairs']],
                            side_known=all(c['side_known'] for c in children),
                            physical_sides=[side for c in children for side in c['physical_sides']],
                            probability_mean=sum(c['length_mm']*c['reference_score'] for c in children)/length,
                            length_mm=length,
                            line_fraction=sum(c['kind'] == 'line' for c in children)/len(children))
            points = np.vstack([s['points'] for s in segments])
            faces = {f for pair in metadata['face_pairs'] for f in pair if f >= 0}
            face_pairs = {tuple(pair) for pair in metadata['face_pairs'] if len(pair) == 2 and min(pair) >= 0}
            self.cache[index] = (decoded, metadata, points, faces, face_pairs)
            self.parent_geometry_builds += 1
        return self.cache[index]

    def duplicate_probabilities(self, parents, pairs):
        if parents is not self.parents:
            self.clear()
            self.parents = parents
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        if np.any(pairs < 0) or np.any(pairs >= len(parents)):
            raise ValueError('Relation pair outside complete input parent pool')
        features = []
        for a, b in pairs:
            first, second = self.entry(a), self.entry(b)
            # ParentRelations.pair reads these five immutable attributes.
            # Its incremental selected-set arrays are unused for pair_features.
            context = ParentRelations.__new__(ParentRelations)
            for column, name in enumerate(('parents', 'metadata', 'points', 'faces', 'face_pairs')):
                setattr(context, name, [first[column], second[column]])
            features.append(pair_features(context, 0, 1))
        features = np.asarray(features, np.float32).reshape(-1, 32)
        self.relation_rows += len(features)
        if not len(features):
            return features, np.empty(0)
        model = self.priors.duplicate
        column = list(model.classes_).index(1)
        return features, model.predict_proba(features)[:, column]
