"""Held input reader with separate selection, diagnostic and final-test roles.

This reader supplies the same complete InputCase tensors as TRAIN. It has no
epoch plan, optimizer interface or training-supervision method. References are
available only after a prediction freeze for that job has been authenticated.
"""
from pathlib import Path

from .server_full_shard_dataset import InputCase
from .sjg_local_runtime_904 import digest, read

ROLES = ('MODEL_SELECTION_ONLY', 'PROTECTED_DIAGNOSTIC_ONLY', 'FINAL_TEST_ONLY')


class EvaluationShardDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.base = self.root / 'server_rebuild/full_dataset_v1'
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
        train_sha = {j['workpiece_sha256'] for j in train['cases']}
        self.jobs, self.versions = {}, {}
        for job in self.registry['cases']:
            if job['job_key'] in self.jobs or job['fit_allowed'] or job['workpiece_sha256'] in train_sha:
                raise ValueError('Duplicate, training-authorized or cross-split input job')
            self.jobs[job['job_key']] = job
            for version in job['versions']:
                key, role = version['key'], version['role']
                if key in self.versions or key not in approved or version['fit_allowed'] or role not in ROLES:
                    raise ValueError('Unapproved or duplicated held version')
                p = approved[key]
                expected = ROLES[[p['selection_allowed'], p['diagnostic_only'], p['final_evaluation_allowed']].index(True)]
                if role != expected or version['split'] != p['assigned_split'] or version['family_id'] != p['family_id']:
                    raise ValueError('A held version changed its authorized use')
                self.versions[key] = (job['job_key'], version)
        if set(self.versions) != set(approved):
            raise ValueError('Complete held registration lost a version')

    def ready_jobs(self, role=None):
        if role is not None and role not in ROLES:
            raise ValueError('Use an explicit held evaluation role')
        return [key for key, job in self.jobs.items()
                if (role is None or any(v['role'] == role for v in job['versions']))
                and (self.shards / key / 'receipt.json').is_file()]

    def input_case(self, job_key):
        job = self.jobs[job_key]
        folder = self.shards / job_key
        receipt = read(folder / 'receipt.json')
        if (receipt['registry_sha256'] != self.registry_sha256
                or receipt['workpiece_sha256'] != job['workpiece_sha256'] or receipt['fit_allowed']):
            raise ValueError('Held input source/registration binding changed')
        if receipt['GT_geometry_reads'] != 0 or receipt['STEP_reads'] != 0 or receipt['screened_by_model']:
            raise ValueError('Held features must be complete unscreened XYZ inputs')
        for name, sha in receipt['files'].items():
            if Path(name).name != name or digest(folder / name) != sha:
                raise ValueError('Held input artifact changed')
        if (digest(folder / 'evaluation_versions.json') != receipt['evaluation_versions_sha256']
                or read(folder / 'evaluation_versions.json')['versions'] != job['versions']):
            raise ValueError('Held input-to-version binding changed')
        return InputCase(folder, receipt)

    def reference(self, version_key, prediction_folder, final_selection=None):
        """Load a reference after authenticated prediction; TEST also needs selection.

        This is a workflow guard, not an independence certificate. A selection
        record's validation evidence must separately be reviewed by the caller.
        """
        job_key, version = self.versions[version_key]
        if version['role'] == 'FINAL_TEST_ONLY' and final_selection is None:
            raise ValueError('Final test references require a model fixed by validation before testing')
        prediction_folder = Path(prediction_folder).resolve()
        prediction_folder.relative_to(self.root)
        freeze = read(prediction_folder / 'prediction_freeze.json')
        pool_file = prediction_folder / 'pool_receipt.json'
        pool = read(pool_file)
        if (freeze['runtime_uses_reference_geometry'] or pool['job_key'] != job_key
                or pool['registry_sha256'] != self.registry_sha256
                or pool['geometry_GT_reads'] != 0 or pool['STEP_reads'] != 0
                or digest(pool_file) != freeze['query_pool_receipt_sha256']
                or digest(prediction_folder / 'selected_parents.json.gz') != freeze['selected_geometry_sha256']):
            raise ValueError('Held reference requires this job\'s completed input-only prediction freeze')
        if version['role'] == 'FINAL_TEST_ONLY':
            selected = read(final_selection)
            if (selected['status'] != 'FINAL_MODEL_SELECTED_VALIDATION_ONLY_BEFORE_TEST'
                    or selected['diagnostic_used_for_selection'] or selected['test_evaluations_for_selection'] != 0
                    or selected['query_checkpoint_sha256'] != pool['query_checkpoint_sha256']
                    or selected['selector_checkpoint_sha256'] != freeze['checkpoint_sha256']):
                raise ValueError('Final test prediction differs from the fixed validation-selected model')
        folder = (self.root / version['label_directory']).resolve()
        folder.relative_to(self.root)
        if digest(folder / 'receipt.json') != version['label_receipt_sha256']:
            raise ValueError('Held qualified reference changed')
        receipt = read(folder / 'receipt.json')
        for name, sha in receipt['artifacts'].items():
            if Path(name).name != name or digest(folder / name) != sha:
                raise ValueError('Held qualified reference artifact changed')
        return dict(role=version['role'], version_key=version_key, family_id=version['family_id'],
                    evaluation_weight=version['family_equal_version_evaluation_weight'],
                    targets=read(folder / 'targets.json'), label_folder=folder,
                    independent_generalization_certified=False)
