"""Immutable full source-domain indices, independent of teacher labels or torch."""
import json
from pathlib import Path
import numpy as np

from .raw_triangle_neural_features_281 import RawTriangleCatalog, file_sha


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CompleteMixedDomainMetadata:
    """Use the same legacy/carrier/RawTriangle ordering as MixedSourceFeatures."""
    def __init__(self, root, source_entry):
        self.root = Path(root)
        directory = self.root / source_entry["feature_directory"]
        producer_path = directory / "receipt.json"
        review_path = directory / "independent_complete_review_attempt1/receipt.json"
        if file_sha(producer_path) != source_entry["producer_receipt_sha256"]:
            raise ValueError("Frozen original source input receipt changed")
        if file_sha(review_path) != source_entry["independent_receipt_sha256"]:
            raise ValueError("Frozen full source input independent review changed")
        producer, review = read(producer_path), read(review_path)
        if review["status"] != "PASS_INDEPENDENT_COMPLETE_SOURCE_CANONICAL_ANCHOR_DOMAIN_CONTEXT_INPUTS":
            raise ValueError("Complete original source input review required")
        if review["producer_receipt_sha256"] != source_entry["producer_receipt_sha256"]:
            raise ValueError("Source review is bound to a different input receipt")
        if producer["source_index"] != review["source_index"] or producer["source_index"] != source_entry["source_index"]:
            raise ValueError("Different original source indices")
        if producer["original_pcd_sha256"] != review["original_pcd_sha256"] or producer["original_pcd_sha256"] != source_entry["original_pcd_sha256"]:
            raise ValueError("Different original PCD identities")
        if producer["label_reads"] or producer["STEP_reads"] or producer["teacher_geometry_or_selected_domains_read"]:
            raise ValueError("Domain registry must be independent of supervision")
        if producer["feature_schema"]["material_channels"]:
            raise ValueError("This source registry has no confirmed material input channels")
        artifacts = producer["artifacts"]
        for name in ("surface_face_ids.npy", "carrier_ids.json"):
            if file_sha(self.root / artifacts[name]["path"]) != artifacts[name]["sha256"]:
                raise ValueError("Immutable original domain metadata changed: " + name)
        faces = np.load(self.root / artifacts["surface_face_ids.npy"]["path"], mmap_mode="r", allow_pickle=False)
        carriers = read(self.root / artifacts["carrier_ids.json"]["path"])
        self.legacy_records = [{"type": "LEGACY_SURFACE", "face_id": int(value)} for value in faces]
        self.legacy_records += [{"type": "RAW_CARRIER", "carrier_id": value} for value in carriers]
        self.legacy_lookup = {key(value): i for i, value in enumerate(self.legacy_records)}
        if len(self.legacy_lookup) != len(self.legacy_records):
            raise ValueError("Duplicate source domains cannot create extra pointer capacity")
        if len(faces) != producer["counts"]["surface_domains"] or len(carriers) != producer["counts"]["carrier_domains"]:
            raise ValueError("Full legacy and original carrier scope required")
        self.raw_triangles = RawTriangleCatalog(self.root, source_entry)
        if self.raw_triangles.receipt["initial_source_input_receipt_sha256"] != source_entry["producer_receipt_sha256"]:
            raise ValueError("RawTriangle registry refers to a different initial source input")
        self.legacy_domain_count = len(self.legacy_records)
        self.total_domain_count = self.legacy_domain_count + self.raw_triangles.count
        self.anchor_count = int(producer["counts"]["canonical_anchors"])
        self.source_index = source_entry["source_index"]

    def index(self, domain):
        if not isinstance(domain, dict):
            raise ValueError("Explicit source-domain metadata required")
        if domain.get("type") == "RAW_TRIANGLE":
            return self.legacy_domain_count + self.raw_triangles.index(domain)
        encoded = key(domain)
        if encoded not in self.legacy_lookup:
            raise ValueError("Loss domain is absent from the unchanged full source registry")
        return self.legacy_lookup[encoded]

    def record(self, index):
        if type(index) is not int or not 0 <= index < self.total_domain_count:
            raise ValueError("Pointer outside the complete original mixed-domain pool")
        if index < self.legacy_domain_count:
            return dict(self.legacy_records[index])
        return self.raw_triangles.record(index - self.legacy_domain_count)
