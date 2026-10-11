"""Read a declared dataset partition through unchanged complete input methods.

Only registration loading selects another dataset directory. InputCase,
SupervisionView, complete epoch planning, held prediction freezes and final-test
reference gates are inherited from the original readers without replacement.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np

from .server_full_shard_dataset import FullShardDataset
from .server_full_evaluation_dataset import EvaluationShardDataset, ROLES
from .sjg_local_runtime_904 import digest, read


def dataset_base(root, base):
    root = Path(root).resolve()
    base = (root / 'server_rebuild/full_dataset_v2') if base is None else Path(base).resolve()
    base.relative_to(root)
    return root, base


class PartitionedFullShardDataset(FullShardDataset):
    def __init__(self, root, base=None):
        self.root, self.base = dataset_base(root, base)
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


class PartitionedEvaluationShardDataset(EvaluationShardDataset):
    def __init__(self, root, base=None):
        self.root, self.base = dataset_base(root, base)
        self.shards = self.base / 'evaluation_input_shards'
        self.registry = read(self.base / 'evaluation_registry.json')
        self.registry_sha256 = digest(self.base / 'evaluation_registry.json')
        if self.registry['fit_allowed'] or not self.registry['test_for_final_evaluation_only']:
            raise ValueError('An evaluation registry cannot authorize fitting or early test scoring')
        for name, sha in self.registry['source_bindings'].items():
            if Path(name).name != name or digest(self.base / name) != sha:
                raise ValueError('Frozen evaluation registration source changed')
        policy = read(self.base / 'model_use_policy.json')
        approved = {v['key']: v for v in policy['examples'] if v['qualified'] and not v['fit_allowed']}
        train = read(self.base / 'training_registry.json')
        train_sha = {job['workpiece_sha256'] for job in train['cases']}
        self.jobs, self.versions = {}, {}
        for job in self.registry['cases']:
            if job['job_key'] in self.jobs or job['fit_allowed'] or job['workpiece_sha256'] in train_sha:
                raise ValueError('Duplicate, training-authorized or cross-split input job')
            self.jobs[job['job_key']] = job
            for version in job['versions']:
                key, role = version['key'], version['role']
                if key in self.versions or key not in approved or version['fit_allowed'] or role not in ROLES:
                    raise ValueError('Unapproved or duplicated held version')
                specification = approved[key]
                expected = ROLES[[specification['selection_allowed'], specification['diagnostic_only'],
                                  specification['final_evaluation_allowed']].index(True)]
                if role != expected or version['split'] != specification['assigned_split'] or version['family_id'] != specification['family_id']:
                    raise ValueError('A held version changed its authorized use')
                self.versions[key] = (job['job_key'], version)
        if set(self.versions) != set(approved):
            raise ValueError('Complete held registration lost a version')
