"""Full qualified TRAIN views, with input-only features and separate supervision.

All finite profiles and all ordered reference parents survive loading. Family
and source identities control sampling/auditing only; they are never columns in
the model input. A complete fit must admit every registered shard and explicitly
handle same-input geometry variants rather than inventing shared negatives.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .sjg_local_runtime_904 import digest, read

GRID = np.array([.125, .375, .625, .875], dtype=np.float64)


def readonly(array):
    array = np.asarray(array)
    array.setflags(write=False)
    return array


class IncompleteTrainingInputs(RuntimeError):
    pass


class InputCase:
    """Frozen input features; this object never opens a supervision file."""
    def __init__(self, folder, receipt):
        self.bindings = dict(receipt['input_bindings'])
        self.input_recipe = dict(schema='FULL_FRAGMENT152_MEAN_MIN_MAX_QUERY_V1',
                                 producer_source_sha256=receipt['source_sha256'],
                                 raw_columns=152, feature_values_preserved=True,
                                 transitive_historical_recipe_equivalence_not_certified=True)
        with np.load(folder / 'fragment_inputs.npz', allow_pickle=False) as arrays:
            self.raw = readonly(arrays['raw_features'])
            self.offsets = readonly(arrays['offsets'])
            self.spans = readonly(arrays['spans'])
            self.canonical = readonly(arrays['canonical'])
            self.representative = readonly(arrays['representative'])
        self.profiles = read(folder / 'profiles.json')
        if self.raw.shape != (2 * len(self.canonical), 152) or not np.isfinite(self.raw).all():
            raise ValueError('Full raw fragment feature contract changed')
        if self.offsets.shape != (len(self.profiles) + 1,) or self.offsets[0] != 0 or self.offsets[-1] != len(self.spans):
            raise ValueError('Finite profile partition changed')
        if np.any(np.diff(self.offsets) < 0) or self.representative.shape != (len(self.spans),):
            raise ValueError('Invalid fragment offsets or aliases')
        if self.spans.ndim != 2 or self.spans.shape[1] != 4 or not np.isfinite(self.spans).all():
            raise ValueError('Invalid original fragment geometry')
        if len(self.canonical) and (np.any(np.diff(self.canonical) <= 0) or self.canonical[0] < 0 or self.canonical[-1] >= len(self.spans)):
            raise ValueError('Canonical input fragment indices changed')
        rows = np.searchsorted(self.canonical, self.representative)
        if np.any(rows >= len(self.canonical)) or not np.array_equal(self.canonical[rows], self.representative):
            raise ValueError('Fragment alias lacks an explicit canonical feature row')
        self.fragment_feature_rows = readonly(rows)
        self.fragment_profiles = readonly(np.searchsorted(self.offsets, np.arange(len(self.spans)), side='right') - 1)
        self.lengths = np.zeros(len(self.profiles), dtype=np.float64)
        for pid, (begin, end) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            spans = self.spans[begin:end, :2]
            if not len(spans):
                continue  # Preserve empty input profiles; never remove targets.
            if spans[0, 0] != 0 or np.any(spans[:, 1] <= spans[:, 0]) or not np.array_equal(spans[1:, 0], spans[:-1, 1]):
                raise ValueError('Fragments do not exactly partition a positive finite carrier')
            self.lengths[pid] = spans[-1, 1]
        self.lengths = readonly(self.lengths)
        self.n = len(self.canonical) * 4  # Original canonical query convention.
        self.all_query_count = len(self.spans) * 4

    def _features(self, fragment, query_index):
        raw_rows = self.fragment_feature_rows[fragment]
        raw = np.asarray(self.raw[np.column_stack((raw_rows * 2, raw_rows * 2 + 1))], dtype=np.float64)
        spans = self.spans[fragment, :2]
        pid = self.fragment_profiles[fragment]
        query = (spans[:, 0] + GRID[query_index] * (spans[:, 1] - spans[:, 0])) / self.lengths[pid]
        features = np.column_stack((raw.mean(1), raw.min(1), raw.max(1), query)).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite query features')
        return features

    @staticmethod
    def _ids(ids, count):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= count):
            raise ValueError('Query IDs outside input-derived range')
        return ids

    def features(self, ids):
        """Exactly the original canonical 457-column fixed query convention."""
        ids = self._ids(ids, self.n)
        return self._features(self.canonical[ids // 4], ids % 4)

    def all_query_features(self, ids):
        """Every actual profile fragment, including aliased finite carriers."""
        ids = self._ids(ids, self.all_query_count)
        return self._features(ids // 4, ids % 4)

    def query_geometry(self, ids):
        ids = self._ids(ids, self.all_query_count)
        fragment = ids // 4
        spans = self.spans[fragment, :2]
        q = spans[:, 0] + GRID[ids % 4] * (spans[:, 1] - spans[:, 0])
        return dict(profile_ids=self.fragment_profiles[fragment], fragment_ids=fragment, positions_mm=q,
                    query_positions_from_input_only=True)

    def features_at(self, pid, positions_mm):
        """Shared features for input-derived grid/event queries on either kind."""
        positions = np.asarray(positions_mm, dtype=np.float64)
        begin, end = self.offsets[pid:pid + 2]
        length = self.lengths[pid]
        if positions.ndim != 1 or not len(self.spans[begin:end]) or not np.isfinite(positions).all():
            raise ValueError('Nonempty finite profile and query positions required')
        if np.any(positions < 0) or np.any(positions > length):
            raise ValueError('Query outside finite input carrier')
        local = np.minimum(np.searchsorted(self.spans[begin:end, 1], positions, side='right'), end - begin - 1)
        rows = self.fragment_feature_rows[begin + local]
        raw = np.asarray(self.raw[np.column_stack((rows * 2, rows * 2 + 1))], dtype=np.float64)
        return np.column_stack((raw.mean(1), raw.min(1), raw.max(1), positions / length)).astype(np.float32)

    def grid_chunks(self, pid, spacing_mm=1.0, chunk_size=2048):
        from .server_grid_feature_stream import grid_feature_chunks
        begin, end = self.offsets[pid:pid + 2]
        yield from grid_feature_chunks(self.lengths[pid], self.spans[begin:end, :2],
                                       self.fragment_feature_rows[begin:end], self.raw,
                                       spacing_mm=spacing_mm, chunk_size=chunk_size)

    def tokens(self, pid):
        if not 0 <= int(pid) < len(self.profiles):
            raise ValueError('Unknown input profile')
        begin, end = self.offsets[pid:pid + 2]
        spans = self.spans[begin:end, :2]
        length = self.lengths[pid]
        if not len(spans):
            return np.empty((0, 459), dtype=np.float32), 0.0
        rows = self.fragment_feature_rows[begin:end]
        raw = np.asarray(self.raw[np.column_stack((rows * 2, rows * 2 + 1))], dtype=np.float64)
        tokens = np.column_stack((raw.mean(1), raw.min(1), raw.max(1), spans / length,
                                  np.full(len(spans), np.log1p(length)))).astype(np.float32)
        return tokens, float(length)

    def curve(self, pid):
        from .sjg_curve_queries_907 import CurveCarrier
        return CurveCarrier(self.profiles[int(pid)]['representation'])


class SupervisionView:
    """Version-specific ordered geometry, partial process labels and raw frames."""
    def __init__(self, folder, version, receipt, compatibility):
        self.version_key = version['key']
        self.family_id = version['family_id']  # Sampling metadata, not features.
        self.family_equal_version_weight = version['family_equal_version_weight']
        self.compatibility_status = compatibility['status']
        self.target = read(folder / 'targets.json')
        self.parents = self.target['parents']
        self.records = read(folder / 'attribute_records.json')
        self.endpoint_evidence = read(folder / 'endpoint_evidence.json')
        ordered_ids = [child['original_child_id'] for parent in self.parents for child in parent['children']]
        if ordered_ids != [row['child_id'] for row in self.records]:
            raise ValueError('Attribute owners do not follow complete ordered geometry')
        if self.target['role'] != 'TRAIN' or receipt['role'] != 'TRAIN':
            raise ValueError('Held supervision cannot enter a TRAIN view')
        with np.load(folder / 'attribute_labels.npz', allow_pickle=False) as arrays:
            self.attribute_arrays = {name: readonly(arrays[name]) for name in arrays.files}
        state = self.attribute_arrays['state']
        if state.shape != (len(ordered_ids), 8):
            raise ValueError('Full child attribute shape changed')
        self.label_known = readonly(state >= 0)
        self.numeric_known = readonly(state == 0)
        self.task_applicable = self.attribute_arrays['required_mask']
        self.required_for_evaluation = self.attribute_arrays['required_mask']
        self.numeric_mm = self.attribute_arrays['exact_mm']
        if not np.isfinite(self.numeric_mm[self.numeric_known]).all() or not np.isnan(self.numeric_mm[~self.numeric_known]).all():
            raise ValueError('Unknown or censored fields were converted into exact distances')
        self.parent_offsets = readonly(np.r_[0, np.cumsum([len(parent['children']) for parent in self.parents])].astype(np.int64))
        if int(self.parent_offsets[-1]) != self.target['original_children']:
            raise ValueError('A complete reference child was lost')
        frames = np.full((len(ordered_ids), 2, 3, 3), np.nan)
        translations = np.full((len(ordered_ids), 2, 3), np.nan)
        frame_known = np.zeros((len(ordered_ids), 2), dtype=bool)
        by_parent = {parent['parent_id']: parent for parent in self.endpoint_evidence['parents']}
        row = 0
        for parent in self.parents:
            children = by_parent[parent['original_parent_id']]['children']
            if [child['child_id'] for child in children] != [child['original_child_id'] for child in parent['children']]:
                raise ValueError('Frame owners or child order changed')
            for child in children:
                for side, name in enumerate(('start', 'end')):
                    frame = child['endpoint_frames'][name]
                    if frame['known']:
                        frames[row, side] = frame['raw_export_axes']
                        translations[row, side] = frame['translation']
                        frame_known[row, side] = True
                row += 1
        if not np.isfinite(frames[frame_known]).all() or not np.isfinite(translations[frame_known]).all():
            raise ValueError('Known export frames are nonfinite')
        self.raw_export_axes = readonly(frames)
        self.frame_translations = readonly(translations)
        self.frame_algebraically_known = readonly(frame_known)
        self.frame_axis_signs_reinterpreted = False
        self.business_frame_semantics_certified = False
        self.physical_background_labels = False
        self.global_physical_parent_count_complete = False
        self.target_sha256 = version['targets_sha256']


class FullShardDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.base = self.root / 'server_rebuild/full_dataset_v1'
        self.shards = self.base / 'input_shards'
        self.registry_sha256 = digest(self.base / 'training_registry.json')
        self.registry = read(self.base / 'training_registry.json')
        policy = read(self.base / 'model_use_policy.json')
        compatibility = read(self.base / 'training_version_compatibility.json')
        split_sha = digest(self.base / 'split_manifest.json')
        if self.registry['split_manifest_sha256'] != split_sha or policy['split_manifest_sha256'] != split_sha:
            raise ValueError('Dataset or model-use split binding changed')
        if compatibility['registry_sha256'] != self.registry_sha256:
            raise ValueError('Same-input supervision audit changed registry')
        if policy['qualification_summary_sha256'] != self.registry['qualification_summary_sha256']:
            raise ValueError('Model-use qualification binding changed')
        self.jobs = {job['job_key']: job for job in self.registry['cases']}
        self.compatibility = {row['job_key']: row for row in compatibility['rows']}
        allowed = {row['key'] for row in policy['examples'] if row['fit_allowed']}
        self.versions = {}
        self.families = defaultdict(list)
        for job in self.jobs.values():
            for version in job['versions']:
                key = version['key']
                if key not in allowed or key in self.versions or version['split'] != 'train':
                    raise ValueError('Duplicate or unapproved training view')
                self.versions[key] = (job['job_key'], version)
                self.families[version['family_id']].append(key)
        if set(self.versions) != allowed:
            raise ValueError('A qualified TRAIN view is missing from the complete registry')
        if not np.isclose(sum(version['family_equal_version_weight'] for _, version in self.versions.values()), 1):
            raise ValueError('Family/version loss mass changed')

    def ready_jobs(self):
        return [key for key in self.jobs if (self.shards / key / 'receipt.json').is_file()]

    def require_complete_inputs(self):
        missing = sorted(set(self.jobs) - set(self.ready_jobs()))
        if missing:
            raise IncompleteTrainingInputs(f'{len(missing)} of {len(self.jobs)} full TRAIN input shards are not ready')

    def input_case(self, job_key):
        job = self.jobs[job_key]
        folder = self.shards / job_key
        receipt = read(folder / 'receipt.json')
        if receipt['registry_sha256'] != self.registry_sha256 or receipt['workpiece_sha256'] != job['workpiece_sha256']:
            raise ValueError('Prepared input registry/source binding changed')
        if receipt['GT_geometry_reads'] != 0 or receipt['STEP_reads'] != 0 or receipt['screened_by_model']:
            raise ValueError('Input shard is not a complete unscreened XYZ input')
        for name, expected in receipt['files'].items():
            if Path(name).name != name or digest(folder / name) != expected:
                raise ValueError('Prepared input file changed')
        if digest(folder / 'supervision_versions.json') != receipt['supervision_versions_sha256']:
            raise ValueError('Input-to-supervision binding changed')
        if read(folder / 'supervision_versions.json')['versions'] != job['versions']:
            raise ValueError('A source version disappeared from the prepared shard')
        return InputCase(folder, receipt)

    def supervision(self, version_key):
        job_key, version = self.versions[version_key]
        folder = (self.root / version['label_directory']).resolve()
        folder.relative_to(self.root.resolve())
        if digest(folder / 'receipt.json') != version['label_receipt_sha256']:
            raise ValueError('Qualified label receipt changed')
        receipt = read(folder / 'receipt.json')
        for name, expected in receipt['artifacts'].items():
            if Path(name).name != name or digest(folder / name) != expected:
                raise ValueError('Qualified label artifact changed')
        return SupervisionView(folder, version, receipt, self.compatibility[job_key])

    def epoch_plan(self, epoch, seed=1031, require_ready=True):
        """Every qualified version exactly once, weighted equally within family."""
        if require_ready:
            self.require_complete_inputs()
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        plan = []
        for family in rng.permutation(sorted(self.families)):
            for key in rng.permutation(sorted(self.families[family])):
                job_key, version = self.versions[key]
                plan.append(dict(version_key=key, job_key=job_key,
                                 loss_weight=version['family_equal_version_weight'],
                                 compatibility_status=self.compatibility[job_key]['status']))
        return plan
