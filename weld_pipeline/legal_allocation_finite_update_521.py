"""All-initial-correct TRAIN protection and finite AdamW acceptance.

The runtime never imports this module. Identities/GT occur only in the TRAIN
pool and loss. Native accepted optimizer steps and rejected proposals differ.
"""
from copy import deepcopy
import gc
import math
import random
import numpy as np
import torch
from weld_pipeline.legacy_incremental_supervision_388 import read, digest
from weld_pipeline.legal_allocation_training_data_520 import ROOT, HERE, prepare_inputs
from weld_pipeline.legacy_incremental_corrective_training_loop_425 import tensor
from weld_pipeline.legacy_legal_allocation_loss_519 import event_loss
from weld_pipeline.finite_plate_attribute_head_523 import attribute_loss
from weld_pipeline.legacy_scale_count_training_loop_488 import analytic_adamw_proposal
from weld_pipeline.reports.goal_contract_training_20261005.score_legacy_full65_frozen_385 import adapt_predictions

TRIAL_FRACTIONS = (1., .5, .25, .125)


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"]); random.setstate(state["python"])


class ProtectionPool:
    def __init__(self, batches, records):
        self.batches, self.records = batches, records

    @torch.no_grad()
    def check(self, events, device="cuda"):
        checked = 0
        failures = []
        for batch in self.batches:
            prior = {k: v.to(device) for k, v in batch["prior"].items()}
            out = events.forward_cached(batch["h"].to(device), prior)
            for row in batch["records"]:
                si, qi = row["candidate"], row["query"]
                n = int(out["count_logits"][si, qi].argmax())
                good = float(out["presence_logits"][si, qi]) >= 0. and n in (row["old_n"], row["target_n"])
                good &= bool(out["closed_logits"][si, qi] >= 0.) == row["closed"]
                gaps = (out["gap_logits"][si, qi, :max(0, n - 1)] >= 0.).cpu().tolist()
                old_topology = n == row["old_n"] and gaps == row["old_gaps"]
                true_topology = n == row["target_n"] and gaps == row["target_gaps"]
                good &= old_topology or true_topology
                if row["direction_semantics_confirmed"]:
                    good &= bool(out["direction_logits"][si, qi] >= 0.) == row["old_direction_positive"]
                maximum = None
                if n > 0 and good:
                    slots = row["old_slots"] if n == row["old_n"] else row["target_slots"]
                    nodes = out["positions_mm"][si, qi, slots]
                    world = nodes[..., None] * nodes.new_tensor(row["direction"])[None, None] + nodes.new_tensor(row["origin"])[None, None]
                    wanted = nodes.new_tensor(row["world_truth"])
                    maximum = float(torch.linalg.vector_norm(world - wanted, dim=-1).max())
                    good &= maximum <= 3.
                checked += len(row["original_child_ids"])
                if not good:
                    failures.append(dict(source_TRAIN_only=row["source"], query=qi,
                        original_child_ids_TRAIN_only=row["original_child_ids"], actual_count=n,
                        old_count=row["old_n"], target_count=row["target_n"],
                        actual_gaps=gaps, maximum_world_endpoint_error_mm=maximum))
            del out, prior
        return dict(pass_all=not failures, checked_initial_correct_child_identities=checked,
            registered_initial_correct_child_identities=self.records,
            failures=failures, tolerance_mm=3., every_source_checked=True,
            whole_line_bidirectional_error_bounded_by_same_chart_endpoint_maximum=True)


def build_protection_pool(model, prior, metadata, progress=None):
    batches, count, migration_max = [], 0, 0.
    per_type = dict(Hline=0, Vline=0)
    for source in range(65):
        sequences, features, base, mapping = prepare_inputs(source, prior, metadata)
        device = next(model.events.parameters()).device
        with torch.no_grad():
            h, initial = model.events.current_frozen_features(tensor(features, device),
                {k: tensor(v, device) for k, v in base.items()})
            migrated = model.events.forward_cached(h, initial)
        folder = (ROOT / metadata["seals"][source]["receipt_path"]).parent
        receipt = read(folder / "receipt.json")
        keys = {k.replace("\\", "/"): v for k, v in receipt["frozen_outputs_sha256"].items()}
        prediction_path = folder / "shadow_Hline/predictions.json"
        assert digest(prediction_path) == keys["shadow_Hline/predictions.json"]
        predictions = read(prediction_path)
        adapted = adapt_predictions(predictions, read(folder / "shadow_Hline/invalid_positive_ledger.json"))
        score_path = HERE / "actual_emission_guard_full65_score_467_attempt1" / ("source_%02d.json" % source)
        expected = metadata["prior_adoption"]["training_only_keeping_identity_bindings_sha256"][str(score_path.relative_to(ROOT))]
        assert digest(score_path) == expected
        matches = read(score_path)["shadow_E_ordered_geometry"]["matches"]
        target_path = ROOT / mapping["geometry_targets_path"]
        assert digest(target_path) == mapping["geometry_targets_sha256"]
        target = read(target_path)
        original = {child["original_child_id"]: (parent, ci, child)
                    for parent in target["parents"] for ci, child in enumerate(parent["children"])}
        by_query = {}
        for match in matches:
            pred = predictions[adapted["parents"][match["predicted_parent_index"]]["original_prediction_index"]]
            if pred["segments"][match["predicted_child_index"]]["kind"] != "line":
                continue  # Original HArc is an independently frozen bypass.
            cid = match["original_child_id"]
            gt_parent, gt_ci, child = original[cid]
            si, qi = int(pred["original_candidate_sequence_index"]), int(pred["learned_parent_query_index"])
            seq = sequences[si]
            direction = np.asarray(seq["direction"], float).copy()
            assert abs(float(np.linalg.norm(direction)) - 1.) < 1e-8
            origin = np.asarray(seq["xyz"][0], float)
            n = len(pred["segments"])
            old_positive = float(initial["direction_logits"][si, qi]) >= 0.
            old_slot = match["predicted_child_index"] if old_positive else n - 1 - match["predicted_child_index"]
            truth = np.asarray(child["world_representation"]["endpoints_mm"], float)
            if float((truth[1] - truth[0]) @ direction) < 0.:
                truth = truth[::-1].copy()
            gt_n = gt_parent["original_child_count"]
            gt_children = gt_parent["children"]
            gt_first = np.asarray(gt_children[0]["original_points_mm"][0], float)
            gt_last = np.asarray(gt_children[-1]["original_points_mm"][-1], float)
            gt_positive = float((gt_last - gt_first) @ direction) > 0.
            target_slot = gt_ci if gt_positive else gt_n - 1 - gt_ci
            mapped = next((r for r in mapping["mapped_parents"] if r["original_parent_index"] == gt_parent["original_parent_index"]), None)
            if mapped is not None:
                gt_gaps = [g == "POSITIVE_GAP" for g in mapped["original_gap_status"]]
                if not gt_positive: gt_gaps = gt_gaps[::-1]
                signed = bool(mapped.get("direction_semantics_confirmed"))
            else:
                ordered = gt_children if gt_positive else gt_children[::-1]
                intervals = [np.sort((np.asarray(c["original_points_mm"])[[0, -1]] - origin) @ direction) for c in ordered]
                gt_gaps = [bool(b[0] > a[1] + 1e-8) for a, b in zip(intervals[:-1], intervals[1:])]
                signed = False
            key = (si, qi)
            row = by_query.setdefault(key, dict(source=source, candidate=si, query=qi,
                old_n=n, target_n=gt_n, old_gaps=(initial["gap_logits"][si, qi, :n - 1] >= 0.).cpu().tolist(),
                target_gaps=gt_gaps, closed=False, old_direction_positive=old_positive,
                direction_semantics_confirmed=signed, origin=origin.tolist(), direction=direction.tolist(),
                original_child_ids=[], old_slots=[], target_slots=[], world_truth=[]))
            # A query spanning multiple distinct original parents cannot switch
            # to one parent's count while silently discarding the other IDs.
            if row["target_n"] != gt_n or row["target_gaps"] != gt_gaps:
                row["target_n"] = n; row["target_gaps"] = row["old_gaps"]; target_slot = old_slot
            row["original_child_ids"].append(cid); row["old_slots"].append(old_slot)
            row["target_slots"].append(target_slot); row["world_truth"].append(truth.tolist())
            old_nodes = initial["reference_effective_positions_mm"][si, qi, old_slot]
            new_nodes = migrated["positions_mm"][si, qi, old_slot]
            migration_max = max(migration_max, float((new_nodes - old_nodes).abs().max()))
            assert migration_max <= 1e-7
            per_type["Hline" if seq["type"] == "horizontal" else "Vline"] += 1
            count += 1
        selected = sorted({r["candidate"] for r in by_query.values()})
        if selected:
            remap = {si: i for i, si in enumerate(selected)}
            for row in by_query.values(): row["candidate"] = remap[row["candidate"]]
            batches.append(dict(h=h[selected].cpu(), prior={k: v[selected].cpu() for k, v in initial.items()}, records=list(by_query.values())))
        if progress: progress(source, count, migration_max)
        del sequences, features, base, h, initial, migrated
        gc.collect(); torch.cuda.empty_cache()
    assert per_type == dict(Hline=1805, Vline=293), per_type
    pool = ProtectionPool(batches, count)
    proof = pool.check(model.events)
    assert proof["pass_all"] and proof["checked_initial_correct_child_identities"] == 2098, proof
    proof.update(initial_correct_type_counts=per_type, migration_maximum_parameter_drift_mm=migration_max,
        initially_lost10_V_and_lost115_H_not_claimed_protected=True,
        original_frozen24_Arc_retained_separately=True, TRAIN_only=True)
    return pool, proof


@torch.no_grad()
def trial_submit(parameters, before, proposal, fraction):
    offset = 0
    for parameter, old in zip(parameters, before):
        end = offset + parameter.numel()
        parameter.copy_((old.double() + proposal[offset:end].reshape(old.shape) * fraction).to(parameter.dtype))
        offset = end
    assert offset == len(proposal)


def finite_accept(optimizer, parameters, check, *, commit=True, before_rng=None):
    before_rng = before_rng or rng_state()
    before = [p.detach().clone() for p in parameters]
    before_optimizer = deepcopy(optimizer.state_dict())
    proposal = analytic_adamw_proposal(optimizer, parameters)
    assert bool(torch.isfinite(proposal).all())
    trials, chosen = [], None
    for fraction in TRIAL_FRACTIONS:
        trial_submit(parameters, before, proposal, fraction)
        try:
            report = check()
        except (AssertionError, ValueError, ArithmeticError) as error:
            report = dict(pass_all=False, explicit_numerical_failure=type(error).__name__ + ": " + str(error))
        trials.append(dict(fraction=fraction, protection=report))
        if report["pass_all"]:
            chosen = fraction; break
    for p, old in zip(parameters, before):
        with torch.no_grad(): p.copy_(old)
    if chosen is None or not commit:
        optimizer.load_state_dict(before_optimizer)
        restore_rng(before_rng)
        return dict(accepted=False if chosen is None else True, committed_optimizer_steps=0,
            rejected_all_four=chosen is None, chosen_fraction=chosen, trials=trials,
            parameters_optimizer_RNG_restored=True, proposal_norm=float(torch.linalg.vector_norm(proposal)))
    optimizer.step()
    native = torch.cat([(p.detach().double() - old.double()).reshape(-1) for p, old in zip(parameters, before)])
    assert bool(torch.isfinite(native).all())
    # Moments/step correspond to the single source AdamW proposal. The accepted
    # finite fraction modifies parameter displacement, a declared constrained
    # update rather than a claim of unconstrained AdamW equivalence.
    trial_submit(parameters, before, native, chosen)
    try:
        final = check()
    except (AssertionError, ValueError, ArithmeticError) as error:
        final = dict(pass_all=False, explicit_numerical_failure=str(error))
    if not final["pass_all"]:
        for p, old in zip(parameters, before):
            with torch.no_grad(): p.copy_(old)
        optimizer.load_state_dict(before_optimizer); restore_rng(before_rng)
        return dict(accepted=False, committed_optimizer_steps=0, native_proposal_step_calls=1,
                    trials=trials, actual_native_proposal_guard_failure=final,
                    parameters_optimizer_RNG_restored=True, stop_stage=True)
    actual = torch.cat([(p.detach().double() - old.double()).reshape(-1) for p, old in zip(parameters, before)])
    assert float(torch.linalg.vector_norm(actual)) > 0., "Zero submission cannot pad the update budget"
    return dict(accepted=True, committed_optimizer_steps=1, native_proposal_step_calls=1,
        chosen_fraction=chosen, trials=trials, final_actual_guard=final,
        proposal_norm=float(torch.linalg.vector_norm(native)), submitted_norm=float(torch.linalg.vector_norm(actual)),
        actual_minus_analytic_maximum=float((native - proposal).abs().max()),
        no_zero_submission=True, constrained_finite_fraction_AdamW=True)


def attribute_backward(module, source, prefix, device):
    available = source[prefix + "_available"]
    chosen = np.flatnonzero(available)
    if not len(chosen): return 0., 0
    target = source[prefix + "_targets"]
    totals = {k: np.asarray(target[k][chosen]).sum(0) for k in ("state_known", "scalar_known", "normal_known")}
    value = 0.
    for a in range(0, len(chosen), 128):
        rows = chosen[a:a + 128]
        output = module(tensor(source[prefix + "_features"][rows], device),
                        [source[prefix + "_plate_hypotheses"][i] for i in rows])
        loss = attribute_loss(output, {k: tensor(v[rows], device) for k, v in target.items()}, totals) * .25
        loss.backward(); value += float(loss.detach())
    return value, len(chosen)


def source_update(model, optimizer, source, pool, device="cuda"):
    initial_rng = rng_state()
    model.train(); optimizer.zero_grad(set_to_none=True)
    base = {k: tensor(v, device) for k, v in source["base"].items()}
    out = model.events(tensor(source["features"], device), base)
    event, detail = event_loss(out, source["parents"], base, tensor(source["background_known"], device),
        tensor(source["group_ids"], device), tensor(source["hard_negative_weights"], device),
        source["original_G"], source["original_G_by_group"])
    event.backward()
    path, path_rows = attribute_backward(model.path_attributes, source, "path", device)
    local, local_rows = attribute_backward(model.attributes, source, "local", device)
    model.verify_frozen()
    clips = {}
    for category in ("H", "V"):
        clips[category] = float(torch.nn.utils.clip_grad_norm_(getattr(model.events, category).parameters(), 5.))
    clips["attributes"] = float(torch.nn.utils.clip_grad_norm_(
        list(model.attributes.parameters()) + list(model.path_attributes.parameters()), 5.))
    assert all(math.isfinite(v) for v in clips.values())
    parameters = model.incremental_parameters()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    acceptance = finite_accept(optimizer, parameters, lambda: pool.check(model.events, device), before_rng=initial_rng)
    model.verify_frozen()
    return dict(losses=dict(event=float(event.detach()), path_attribute=path, local_attribute=local),
        preclip_norms=clips, acceptance=acceptance, optimizer_steps=acceptance["committed_optimizer_steps"],
        original_G=source["original_G"], original_E16_records=source["original_E16_records"],
        original_local_points=source["original_local_points"], available_whole_path_rows=path_rows,
        available_local_reference_rows=local_rows, representation_miss_children=source["representation_miss_children"],
        training_correspondence_failure_counts=detail["training_correspondence_failure_counts"],
        all_initial_correct_identities_checked=2098, attribute_gradients_cannot_change_event_geometry=True,
        formal_E_records_created=0)
