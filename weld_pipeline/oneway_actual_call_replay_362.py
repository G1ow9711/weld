"""Write-only actual-operation audit and bounded original-environment replay.

The prediction worker never reads this stream or receives reference feedback.
Replay uses complete original operation shapes, all legal slots and the frozen
decoder. Each record has a fresh Pickler, so its memo cannot retain tensor or
array history. Only scalar batch metadata survives carrier release.
"""
from collections import Counter
from contextlib import AbstractContextManager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil

import numpy as np
import torch

from . import streamed_role_prediction_349 as prediction
from .carrier_observation_sequences_328 import OriginalXYZObservationProvider
from .gate_checked_consuming_decoder_347 import decode_gate_checked_consuming_heads
from .streamed_role_gate_347 import (scan_complete_child_start_gate,
    materialize_complete_heads_after_gate, empty_gate_decoded_result)
from .live_json_snapshot_282 import write_json_snapshot as dump

VERSION = "ONEWAY_ACTUAL_CALL_INPUTS_AND_BYTE_EXACT_DEFAULT_REPLAY_362_v1"
ARRAY_HASH_ROWS = 64
FIXED_CARRIERS_PER_BATCH = 1024


def array_signature(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu" and not value.requires_grad
        data = value.detach().numpy()
        stride = list(value.stride())
        dtype, device = str(value.dtype), str(value.device)
    elif isinstance(value, np.ndarray):
        data = value
        stride = [v // data.dtype.itemsize for v in data.strides]
        dtype, device = str(data.dtype), "numpy_cpu"
    else:
        raise TypeError(type(value))
    h = hashlib.sha256()
    if data.ndim == 0:
        if not bool(np.isfinite(data)):
            raise ValueError("NONFINITE_RAW_OPERATION_VALUE")
        h.update(data.tobytes(order="C"))
    else:
        for first in range(0, len(data), ARRAY_HASH_ROWS):
            part = data[first:first + ARRAY_HASH_ROWS]
            if not bool(np.isfinite(part).all()):
                raise ValueError("NONFINITE_RAW_OPERATION_VALUE")
            h.update(part.tobytes(order="C"))
    return dict(dtype=dtype, device=device, shape=list(data.shape), stride=stride,
        logical_dtype_bytes_sha256=h.hexdigest())


def signature(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return array_signature(value)
    if isinstance(value, dict):
        return {key: signature(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [signature(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NONFINITE_SCALAR_OPERATION_VALUE")
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    raise TypeError("Unregistered audit value: " + str(type(value)))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False)


def scientific_digest(value):
    return hashlib.sha256(canonical(signature(value)).encode("utf-8")).hexdigest()


def layout_signature(layout):
    return dict(carrier_ordinal=int(layout.carrier.ordinal), rows=int(layout.rows),
        legal_slots=int(layout.carrier.slot_count),
        geometry_identity=layout.carrier.geometry_identity,
        aliases_in_original_order=list(layout.carrier.aliases),
        carrier_summary=signature(layout.carrier.summary()),
        arrays={key: array_signature(value) for key, value in vars(layout).items()
            if isinstance(value, np.ndarray)})


class OneWaySnapshotWriter:
    def __init__(self, directory, project_root, minimum_disk_bytes, expected_carriers,
                 software_control=False):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.project_root = Path(project_root)
        self.minimum_disk_bytes = int(minimum_disk_bytes)
        self.expected_carriers = int(expected_carriers)
        self.software_control = bool(software_control)
        self.stream = None
        self.batch_rows = []
        self.kind_counts = Counter()
        self.total_bytes = 0
        self.disk_plan = None
        self.current_ordinal = None
        self.completed_carriers = 0
        self.next_query = 0
        self.pending_decoded_ledger = None
        self.context_recorded = False
        self.neural_calls = Counter()

    def register_generated(self, carriers):
        assert self.disk_plan is None and len(carriers) == self.expected_carriers
        complete_rows = sum(c.cell_count + 2 + len(c.structural_events_mm) for c in carriers)
        complete_slots = sum(c.slot_count for c in carriers)
        halo_rows = sum((c.cell_count + 2 + len(c.structural_events_mm)) +
            4 * (math.ceil((c.cell_count + 2 + len(c.structural_events_mm)) / 8192) - 1)
            for c in carriers)
        # Every carrier may repeat the original observation blocks for full heads.
        observation_bound = 2 * halo_rows * (64 * 4 + 8)
        layout_bound = complete_rows * 49
        metadata_reserve = max(512 * 1024**2, len(carriers) * 65536)
        budget = observation_bound + layout_bound + metadata_reserve
        free = shutil.disk_usage(self.project_root).free
        if free - budget < self.minimum_disk_bytes:
            raise ValueError("REGISTERED_COMPLETE_AUDIT_DISK_BUDGET_NOT_AVAILABLE")
        self.disk_plan = dict(version=VERSION, complete_original_carriers=len(carriers),
            complete_layout_rows=complete_rows, complete_original_legal_slots=complete_slots,
            complete_role_plus_possible_full_head_halo_rows_upper_bound=2 * halo_rows,
            feature_width=64, observation_bytes_upper_bound=observation_bound,
            layout_array_bytes_upper_bound=layout_bound, metadata_and_alias_bytes_reserved=metadata_reserve,
            registered_stream_bytes_budget=budget, disk_free_before_first_observation_bytes=free,
            minimum_remaining_disk_bytes=self.minimum_disk_bytes,
            hard_budget_failure_preserves_partial_stream_and_is_not_success=True,
            fixed_carriers_per_batch=FIXED_CARRIERS_PER_BATCH,
            all_batches_fixed_before_first_original_observation=True,
            software_control=self.software_control)
        dump(self.directory / "disk_budget_registered_before_observations.json", self.disk_plan)

    def record_context(self, source, tokens, context):
        assert self.disk_plan is not None and not self.context_recorded
        packet = dict(version=VERSION, tokens=tokens.detach().numpy(),
            token_signature=array_signature(tokens), context=signature(context),
            source_buffer_signatures={key: array_signature(value)
                for key, value in prediction.source_numeric_buffers(source).items()},
            software_control=self.software_control)
        with (self.directory / "actual_context_snapshot.pkl").open("wb") as stream:
            pickle.dump(packet, stream, protocol=5)
        self.context_recorded = True

    def record_software_context(self, tokens):
        assert self.software_control and self.disk_plan is not None and not self.context_recorded
        with (self.directory / "actual_context_snapshot.pkl").open("wb") as stream:
            pickle.dump(dict(version=VERSION, tokens=tokens.detach().numpy(),
                token_signature=array_signature(tokens), context={"original_points_consumed": 0},
                source_buffer_signatures={}, software_control=True), stream, protocol=5)
        self.context_recorded = True

    def write(self, kind, payload):
        assert self.stream is not None and self.disk_plan is not None
        before = self.stream.tell()
        pickle.dump(dict(kind=kind, payload=payload), self.stream, protocol=5)
        self.total_bytes += self.stream.tell() - before
        self.kind_counts[kind] += 1
        if self.total_bytes > self.disk_plan["registered_stream_bytes_budget"]:
            raise ValueError("COMPLETE_ONEWAY_STREAM_EXCEEDED_PRE_REGISTERED_DISK_BUDGET")

    def close_batch(self):
        if self.stream is None:
            return
        row = self.batch_rows[-1]
        self.write("BATCH_END", dict(first_ordinal=row["first_ordinal"],
            last_ordinal=self.completed_carriers - 1, next_query=self.next_query))
        row.update(last_ordinal=self.completed_carriers - 1,
            complete_carriers=self.completed_carriers - row["first_ordinal"],
            end_query_index=self.next_query, actual_stream_bytes=self.stream.tell())
        self.stream.flush()
        self.stream.close()
        self.stream = None

    def observations(self, layout, first, last, features, groups):
        ordinal = int(layout.carrier.ordinal)
        if ordinal != self.current_ordinal:
            assert ordinal == self.completed_carriers and self.pending_decoded_ledger is None
            if ordinal % FIXED_CARRIERS_PER_BATCH == 0:
                self.close_batch()
                if shutil.disk_usage(self.project_root).free < self.minimum_disk_bytes:
                    raise ValueError("ORIGINAL_REMAINING_DISK_FLOOR")
                path = self.directory / f"batch_{ordinal // FIXED_CARRIERS_PER_BATCH:04d}.pkl"
                self.stream = path.open("wb")
                self.batch_rows.append(dict(file=path.name, first_ordinal=ordinal,
                    start_query_index=self.next_query))
            self.current_ordinal = ordinal
            self.write("CARRIER_BEGIN", dict(layout=layout, layout_signature=layout_signature(layout),
                first_query=self.next_query))
        assert last > first and features.shape == (last - first, 64) and groups.shape == (last - first,)
        self.write("OBSERVATIONS", dict(carrier_ordinal=ordinal, first=first, last=last,
            features=features, groups=groups, features_signature=array_signature(features),
            groups_signature=array_signature(groups)))

    def operation(self, label, args, result):
        self.write("NN_CALL", dict(label=label, input_signatures=signature(args),
            output_signature=signature(result)))
        self.neural_calls[label] += 1

    def decode_input(self, gate, layout, heads, first_query):
        self.write("DECODE_IN", dict(carrier_ordinal=int(layout.carrier.ordinal),
            role_gate=asdict(gate), first_query=int(first_query),
            complete_head_signatures=None if heads is None else signature(heads)))

    def decode_output(self, result):
        ledger = dict(selected_start_slots=int(result["selected_start_slots"]),
            total_prediction_children=int(result["total_prediction_children"]),
            next_query_index=int(result["next_query_index"]),
            valid_children=sum(p["child_count"] for p in result["parents"]),
            counted_invalid_children=sum(p["counted_false_children"] for p in result["invalid_predictions"]))
        assert ledger["selected_start_slots"] == ledger["total_prediction_children"]
        assert ledger["valid_children"] + ledger["counted_invalid_children"] == ledger["selected_start_slots"]
        assert ledger["next_query_index"] - self.next_query == ledger["selected_start_slots"]
        self.write("DECODE_OUT", dict(scientific_output_sha256=scientific_digest(result), ledger=ledger))
        self.pending_decoded_ledger = ledger

    def carrier_released(self, values):
        assert self.pending_decoded_ledger is not None
        assert values["carrier_ordinal"] == self.current_ordinal == self.completed_carriers
        assert values["complete_carriers"] == self.completed_carriers + 1
        assert values["cumulative_selected_start_outputs"] == self.pending_decoded_ledger["next_query_index"]
        self.next_query = self.pending_decoded_ledger["next_query_index"]
        self.write("CARRIER_END", dict(carrier_ordinal=self.current_ordinal,
            next_query=self.next_query, ledger=self.pending_decoded_ledger))
        self.completed_carriers += 1
        self.current_ordinal = None
        self.pending_decoded_ledger = None
        self.stream.flush()

    def finish(self):
        assert self.completed_carriers == self.expected_carriers
        assert self.pending_decoded_ledger is None and self.context_recorded
        self.close_batch()
        for row in self.batch_rows:
            h = hashlib.sha256()
            with (self.directory / row["file"]).open("rb") as stream:
                while True:
                    part = stream.read(1024 * 1024)
                    if not part:
                        break
                    h.update(part)
            row["sha256"] = h.hexdigest()
        receipt = dict(version=VERSION, status="COMPLETE_WRITE_ONLY_ACTUAL_OPERATION_STREAM_FROZEN",
            complete_carriers=self.completed_carriers, selected_start_outputs=self.next_query,
            disk_plan=self.disk_plan, actual_stream_bytes=self.total_bytes,
            record_counts=dict(self.kind_counts), complete_neural_call_counts=dict(self.neural_calls),
            batches=self.batch_rows, no_operation_tensor_or_array_history_retained=True,
            original_computational_shapes_unchanged=True, scientific_prediction_reference_reads=0,
            reference_feedback_into_scientific_prediction=0, software_control=self.software_control,
            independent_reference_not_yet_run=True, full_contract_pass=False)
        dump(self.directory / "receipt.json", receipt)
        return receipt


class BoundedReplayReader:
    def __init__(self, path, batch):
        self.stream = Path(path).open("rb")
        self.batch = batch
        self.neural_calls = Counter()
        self.complete_carriers = 0

    def pop(self, expected):
        record = pickle.load(self.stream)
        if record["kind"] != expected:
            raise ValueError("COMPLETE_OPERATION_ORDER_MISMATCH: " + str((expected, record["kind"])))
        return record["payload"]

    def begin(self):
        packet = self.pop("CARRIER_BEGIN")
        layout = packet["layout"]
        assert layout_signature(layout) == packet["layout_signature"]
        assert layout.carrier.ordinal == self.batch["first_ordinal"] + self.complete_carriers
        return layout, packet["first_query"]

    def observations(self, layout, first=0, last=None):
        last = layout.rows if last is None else last
        packet = self.pop("OBSERVATIONS")
        assert (packet["carrier_ordinal"], packet["first"], packet["last"]) == (layout.carrier.ordinal, first, last)
        features, groups = packet["features"], packet["groups"]
        assert array_signature(features) == packet["features_signature"]
        assert array_signature(groups) == packet["groups_signature"]
        return features, groups

    def operation(self, label, args, result):
        expected = self.pop("NN_CALL")
        observed = dict(label=label, input_signatures=signature(args), output_signature=signature(result))
        if canonical(expected) != canonical(observed):
            raise ValueError("ACTUAL_CROSS_ENVIRONMENT_BYTE_SHAPE_OR_STRIDE_MISMATCH: " + label)
        self.neural_calls[label] += 1

    def decode_input(self, gate, layout, heads, first_query):
        expected = self.pop("DECODE_IN")
        actual = dict(carrier_ordinal=int(layout.carrier.ordinal), role_gate=asdict(gate),
            first_query=int(first_query), complete_head_signatures=None if heads is None else signature(heads))
        if canonical(expected) != canonical(actual):
            raise ValueError("COMPLETE_ROLE_GATE_OR_FULL_HEAD_BUFFER_MISMATCH")

    def decode_output(self, result):
        expected = self.pop("DECODE_OUT")
        if scientific_digest(result) != expected["scientific_output_sha256"]:
            raise ValueError("FULL_POINTER_LINK_PROJECTION_INVALID_OR_QUERY_OUTPUT_MISMATCH")
        assert result["next_query_index"] == expected["ledger"]["next_query_index"]
        self.pending_ledger = expected["ledger"]

    def end(self, layout, result):
        expected = self.pop("CARRIER_END")
        assert expected == dict(carrier_ordinal=int(layout.carrier.ordinal),
            next_query=int(result["next_query_index"]), ledger=self.pending_ledger)
        self.complete_carriers += 1

    def finish(self, next_query):
        expected = self.pop("BATCH_END")
        assert expected == dict(first_ordinal=self.batch["first_ordinal"],
            last_ordinal=self.batch["last_ordinal"], next_query=next_query)
        assert self.complete_carriers == self.batch["complete_carriers"]
        assert not self.stream.read(1)
        self.stream.close()


class ActualCallAudit(AbstractContextManager):
    """Observe returned values, preserving original methods and no forward hooks."""
    def __init__(self, model, sink, writer=False, stage_profile=None):
        self.model, self.sink, self.writer = model, sink, bool(writer)
        self.original_stage_profile = stage_profile or (lambda **values: None)
        self.restorations = []

    def patch(self, obj, name, replacement):
        was_instance_attribute = name in vars(obj)
        previous = getattr(obj, name)
        setattr(obj, name, replacement)
        self.restorations.append((obj, name, previous, was_instance_attribute))

    def __enter__(self):
        model = self.model
        for name in ("encode_sequence", "slot_embeddings", "heads", "pointer_logits", "relation_logits"):
            original = getattr(model, name)
            def wrapper(*args, _original=original, _name=name, **kwargs):
                assert not kwargs
                result = _original(*args)
                self.sink.operation(_name, args, result)
                return result
            self.patch(model, name, wrapper)
        original_roles = model.roles.forward
        def roles(*args, **kwargs):
            assert not kwargs
            result = original_roles(*args)
            self.sink.operation("roles", args, result)
            return result
        self.patch(model.roles, "forward", roles)
        if self.writer:
            original_context = model.context.forward
            def context(source):
                tokens, meta = original_context(source)
                self.sink.record_context(source, tokens, meta)
                return tokens, meta
            self.patch(model.context, "forward", context)
            original_observations = OriginalXYZObservationProvider.observations
            def observations(provider, layout, first=0, last=None):
                last = layout.rows if last is None else last
                features, groups = original_observations(provider, layout, first, last)
                self.sink.observations(layout, first, last, features, groups)
                return features, groups
            self.patch(OriginalXYZObservationProvider, "observations", observations)
            original_build = prediction.build_compact_inference_carriers
            def build(records):
                carriers = original_build(records)
                self.sink.register_generated(carriers)
                return carriers
            self.patch(prediction, "build_compact_inference_carriers", build)
            self.patch(prediction, "decode_gate_checked_consuming_heads", self.decode)
            self.patch(prediction, "empty_gate_decoded_result", self.empty)
        return self

    def decode(self, model, carrier, layout, heads, gate, first_query=0):
        self.sink.decode_input(gate, layout, heads, first_query)
        result = decode_gate_checked_consuming_heads(model, carrier, layout, heads, gate, first_query)
        self.sink.decode_output(result)
        return result

    def empty(self, gate, layout, first_query=0):
        self.sink.decode_input(gate, layout, None, first_query)
        result = empty_gate_decoded_result(gate, layout, first_query)
        self.sink.decode_output(result)
        return result

    def profile(self, **values):
        self.original_stage_profile(**values)
        if self.writer and values["stage"] == "CARRIER_RELEASED" and values["edge"] == "END":
            self.sink.carrier_released(values)

    def __exit__(self, *error):
        for obj, name, previous, was_instance_attribute in reversed(self.restorations):
            if was_instance_attribute:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self.restorations.clear()
        if self.writer and self.sink.stream is not None and error[0] is not None:
            self.sink.stream.flush()
            self.sink.stream.close()
            self.sink.stream = None
        return False


def replay_complete_batch(model, tokens, reader, progress=None):
    parents, invalid, calls, next_query = [], [], 0, reader.batch["start_query_index"]
    with torch.no_grad(), ActualCallAudit(model, reader) as audit:
        for offset in range(reader.batch["complete_carriers"]):
            layout, first_query = reader.begin()
            assert first_query == next_query
            gate = scan_complete_child_start_gate(model, tokens, reader, layout,
                compare_original_whole_sigmoid=True)
            calls += gate.role_block_forwards
            if gate.selected_count:
                heads, full_calls = materialize_complete_heads_after_gate(model, tokens, reader, layout, gate)
                calls += full_calls
                decoded = audit.decode(model, layout.carrier, layout, heads, gate, first_query)
                assert not heads
                del heads
            else:
                decoded = audit.empty(gate, layout, first_query)
            reader.end(layout, decoded)
            parents.extend(decoded["parents"])
            invalid.extend(decoded["invalid_predictions"])
            next_query = decoded["next_query_index"]
            ordinal = layout.carrier.ordinal
            del layout, gate, decoded
            if progress and (offset % 32 == 0 or offset + 1 == reader.batch["complete_carriers"]):
                progress(complete_carriers=offset + 1, last_complete_ordinal=ordinal,
                    actual_encoder_calls=calls, current_next_query_index=next_query)
    reader.finish(next_query)
    return dict(parents=parents, invalid_predictions=invalid, next_query_index=next_query,
        complete_carriers=reader.complete_carriers, actual_encoder_calls=calls,
        complete_neural_call_counts=dict(reader.neural_calls),
        default_environment_replay_is_not_uninterrupted_original_PCD_prediction=True,
        scientific_candidate_received_reference_feedback=False, full_contract_pass=False)
