"""Training-only source observations, sequential disk blocks and provenance.

Runtime prediction MUST NOT consume this cache. Free evaluation rebuilds
observations with OriginalXYZObservationProvider from its actual source.
No label or GT assignment is accepted by the cache builder.
"""
from pathlib import Path
import os
import numpy as np
from .analytic_carrier_slots_327 import build_carriers
from .carrier_observation_sequences_328 import OriginalXYZObservationProvider, SequenceLayout, FEATURE_WIDTH, FEATURE_BLOCK_ROWS
from .primitive_incidence_209 import array_sha
from .live_json_snapshot_282 import write_json_snapshot as dump
from .world_set_training_285 import utc, file_sha, read

VERSION = "TRAINING_ONLY_ORIGINAL_XYZ_COMPLETE_SEQUENCE_CACHE_328_v1"


def write_training_observations(source, bank_records, output, source_bindings, progress=None):
    output = Path(output)
    output.mkdir(exist_ok=False)
    report = progress or (lambda **values: None)
    carriers = build_carriers(bank_records)
    provider = OriginalXYZObservationProvider(source)
    before = array_sha(source.xyz_mm)
    features_path, groups_path = output / "pure_observations_f32.bin", output / "nearby_original_groups_u16.bin"
    rows, slots = 0, 0
    indices = []
    with features_path.open("wb") as features_file, groups_path.open("wb") as groups_file:
        for ordinal, carrier in enumerate(carriers):
            layout = SequenceLayout.from_carrier(carrier)
            record = dict(carrier_ordinal=ordinal, row_offset=rows, row_count=layout.rows,
                complete_slots=carrier.slot_count, exact_geometry_identity=carrier.geometry_identity)
            for first in range(0, layout.rows, FEATURE_BLOCK_ROWS):
                last = min(first + FEATURE_BLOCK_ROWS, layout.rows)
                values, groups = provider.observations(layout, first, last)
                assert values.shape == (last - first, FEATURE_WIDTH) and groups.shape == (last - first,)
                assert (groups >= 0).all() and (groups < len(provider.available_groups)).all()
                features_file.write(np.asarray(values, dtype="<f4").tobytes())
                groups_file.write(np.asarray(groups, dtype="<u2").tobytes())
            indices.append(record)
            rows += layout.rows; slots += carrier.slot_count
            if ordinal % 20 == 0 or ordinal + 1 == len(carriers):
                report(complete_carriers=ordinal + 1, original_carriers=len(carriers), complete_observation_rows=rows, complete_boundary_slots=slots)
        features_file.flush(); os.fsync(features_file.fileno())
        groups_file.flush(); os.fsync(groups_file.fileno())
    assert features_path.stat().st_size == rows * FEATURE_WIDTH * 4 and groups_path.stat().st_size == rows * 2
    assert array_sha(source.xyz_mm) == before
    receipt = dict(utc=utc(), version=VERSION, status="COMPLETE_TRAINING_ONLY_PURE_XYZ_ALL_SEQUENCE_OBSERVATIONS",
        feature_width=FEATURE_WIDTH, original_XYZ=len(source.xyz_mm), original_array_sha256=before,
        exact_alias_capacity_domains=len(carriers), total_rows=rows, total_slots=slots, carrier_indices=indices,
        source_code_sha256=source_bindings, file_sha256={features_path.name:file_sha(features_path), groups_path.name:file_sha(groups_path)},
        target_reads_for_feature_generation=0, STEP_reads=0, every_declared_boundary_slot_retained=True,
        runtime_may_read_this_cache=False, original_source_context_not_cached=True, actual_optimizer_updates=0, full_contract_pass=False)
    dump(output / "receipt.json", receipt)
    return receipt


class TrainingObservationReader:
    def __init__(self, directory, source, carriers, expected_bindings):
        self.directory = Path(directory)
        self.receipt = read(self.directory / "receipt.json")
        if self.receipt["version"] != VERSION or self.receipt["runtime_may_read_this_cache"] or self.receipt["target_reads_for_feature_generation"] or self.receipt["STEP_reads"]:
            raise ValueError("Only the registered training-only pure XYZ cache may be consumed")
        if self.receipt["source_code_sha256"] != expected_bindings or self.receipt["original_array_sha256"] != array_sha(source.xyz_mm):
            raise ValueError("Observation code or original XYZ provenance changed")
        if self.receipt["exact_alias_capacity_domains"] != len(carriers):
            raise ValueError("Complete finite carrier scope changed")
        for name, expected in self.receipt["file_sha256"].items():
            if file_sha(self.directory / name) != expected:
                raise ValueError("Training observation bytes changed")
        self.features = (self.directory / "pure_observations_f32.bin").open("rb")
        self.groups = (self.directory / "nearby_original_groups_u16.bin").open("rb")
        self.carriers = carriers

    def observations(self, ordinal, first=0, last=None):
        record = self.receipt["carrier_indices"][ordinal]
        carrier = self.carriers[ordinal]
        if record["carrier_ordinal"] != ordinal or record["exact_geometry_identity"] != carrier.geometry_identity:
            raise ValueError("Cache rows were assigned to a different carrier")
        last = record["row_count"] if last is None else last
        if not 0 <= first < last <= record["row_count"]:
            raise ValueError("Valid complete sequence block required")
        offset, count = record["row_offset"] + first, last - first
        self.features.seek(offset * FEATURE_WIDTH * 4)
        self.groups.seek(offset * 2)
        values = np.frombuffer(self.features.read(count * FEATURE_WIDTH * 4), dtype="<f4").reshape(count, FEATURE_WIDTH).copy()
        groups = np.frombuffer(self.groups.read(count * 2), dtype="<u2").astype(np.int64)
        if len(groups) != count or not np.isfinite(values).all():
            raise ValueError("Truncated or nonfinite observation block")
        return values, groups

    def close(self):
        self.features.close(); self.groups.close()
