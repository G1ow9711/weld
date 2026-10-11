"""Observed RawTriangle features; metadata never becomes a neural ID channel."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from .pure_pcd_initial_xyz_270 import initial_xyz_features

VERSION = "COMPLETE_ORIGINAL_RAW_TRIANGLE_NEURAL_DOMAIN_FEATURES_279_v1"
CHANNELS = (["centroid_" + x for x in "xyz"]
    + [edge + "_" + x for edge in ("edge_01", "edge_02") for x in "xyz"]
    + ["scaled_gram_11", "scaled_gram_12", "scaled_gram_22"]
    + ["computational_scaled_oriented_cross_" + x for x in "xyz"]
    + ["observed_face_feature_" + str(i) for i in range(21)])
METADATA = ("face_id", "triangle_ordinal", "canonical_centroid_anchor_id", "raw_a", "raw_b", "raw_c")

def schema():
    return dict(version=VERSION, channels=CHANNELS, neural_dtype="float32", metadata_columns=list(METADATA),
        source_metadata_or_ids_are_neural_channels=False,
        exact_world_geometry="unchanged original float64 raw vertices and reviewed RawTriangleChart",
        candidate_scope="every original triangle occurrence in the immutable source registry",
        aliases_add_role_capacity=False, computational_cross_is_signed_material_normal=False,
        source_context="mean of every original RawTriangle feature; no teacher selection",
        GT_STEP_or_material_labels_read=False, actual_optimizer_updates=0, full_contract_pass=False)

def file_sha(path):
    h=sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def triangle_features(raw, metadata, surface_features, face_index, normalizer):
    metadata=np.asarray(metadata)
    if metadata.ndim!=2 or metadata.shape[1]!=6 or metadata.dtype!=np.int64:
        raise ValueError("All original face/triangle/centroid/raw identities required")
    if (metadata[:,3:]<0).any() or (metadata[:,3:]>=len(raw)).any():
        raise ValueError("Original raw triangle references outside bound point source")
    points=np.asarray(raw[metadata[:,3:]],dtype=np.float64)
    scale=normalizer["isotropic_scale_mm"]
    if not np.isfinite(points).all() or not np.isfinite(scale) or scale<=0:
        raise ValueError("Finite unchanged source coordinates and training-only scale required")
    first=(points[:,1]-points[:,0])/scale
    second=(points[:,2]-points[:,0])/scale
    gram=np.column_stack((np.einsum("ij,ij->i",first,first),np.einsum("ij,ij->i",first,second),
                          np.einsum("ij,ij->i",second,second)))
    cross=np.cross(first,second)
    face_rows=np.asarray([face_index[int(v)] for v in metadata[:,0]],dtype=np.int64)
    out=np.column_stack((initial_xyz_features(points.mean(axis=1),normalizer),first,second,gram,cross,
                         np.asarray(surface_features[face_rows],dtype=np.float64))).astype(np.float32)
    if out.shape!=(len(metadata),len(CHANNELS)) or not np.isfinite(out).all():
        raise ValueError("Fixed finite RawTriangle feature schema required; no dropped rows")
    return out

class RawTriangleCatalog:
    """Memory mapped complete domain pool, including every occurrence alias."""
    def __init__(self, root, entry, *, require_independent_review=True):
        self.root=Path(root)
        self.directory=self.root/entry["raw_triangle_feature_directory"]
        receipt_path=self.directory/"receipt.json"
        if file_sha(receipt_path)!=entry["raw_triangle_receipt_sha256"]:
            raise ValueError("Frozen complete RawTriangle feature receipt changed")
        receipt=json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        if receipt["original_pcd_sha256"]!=entry["original_pcd_sha256"]:
            raise ValueError("RawTriangle features and initial source refer to different original PCDs")
        if receipt["label_reads"] or receipt["STEP_reads"] or receipt["teacher_selected_triangle_reads"]:
            raise ValueError("RawTriangle initial input cannot read truth or teacher-selected geometry")
        if require_independent_review:
            review_path=self.directory/"independent_complete_review_attempt1/receipt.json"
            if file_sha(review_path)!=entry["raw_triangle_independent_receipt_sha256"]:
                raise ValueError("Complete RawTriangle independent review changed")
            review=json.loads(review_path.read_text(encoding="utf-8-sig"))
            if review["producer_receipt_sha256"]!=entry["raw_triangle_receipt_sha256"] or review["status"]!="PASS_INDEPENDENT_EVERY_ORIGINAL_RAW_TRIANGLE_NEURAL_DOMAIN_ROW":
                raise ValueError("Complete independent RawTriangle input review required")
        self.arrays={}
        for name, artifact in receipt["artifacts"].items():
            path=self.root/artifact["path"]
            if file_sha(path)!=artifact["sha256"]:
                raise ValueError("Immutable RawTriangle feature artifact changed: "+name)
            self.arrays[name]=np.load(path,mmap_mode="r",allow_pickle=False)
        self.features=self.arrays["raw_triangle_features.npy"]
        self.metadata=self.arrays["raw_triangle_metadata.npy"]
        self.context=self.arrays["complete_raw_triangle_context.npy"]
        self.count=int(receipt["all_original_triangles"])
        if self.features.shape!=(self.count,len(CHANNELS)) or self.features.dtype!=np.float32:
            raise ValueError("Every original RawTriangle must have its own complete feature row")
        if self.metadata.shape!=(self.count,6) or self.metadata.dtype!=np.int64 or self.context.shape!=(len(CHANNELS),):
            raise ValueError("Frozen source metadata and whole-RawTriangle context required")
        self.face_ranges={int(v["face_id"]):(int(v["start_row"]),int(v["triangle_count"])) for v in receipt["ordered_face_ranges"]}
        if sum(v[1] for v in self.face_ranges.values())!=self.count:
            raise ValueError("Complete face ranges must preserve every original triangle occurrence")
        self.receipt=receipt

    def index(self, domain):
        if set(domain)!={"type","face_id","triangle_ordinal"} or domain["type"]!="RAW_TRIANGLE":
            raise ValueError("Exact observed RawTriangle occurrence pointer required")
        if type(domain["face_id"]) is not int or type(domain["triangle_ordinal"]) is not int:
            raise ValueError("Integer original face and triangle occurrence identities required")
        start,count=self.face_ranges[domain["face_id"]]
        ordinal=domain["triangle_ordinal"]
        if not 0<=ordinal<count:
            raise ValueError("Pointer outside original RawTriangle occurrence pool")
        row=start+ordinal
        if tuple(map(int,self.metadata[row,:2]))!=(domain["face_id"],ordinal):
            raise ValueError("Original triangle ordering or pointer identity changed")
        return row

    def record(self, row):
        if type(row) is not int or not 0<=row<self.count:
            raise ValueError("Domain row outside complete original RawTriangle pool")
        face,ordinal=self.metadata[row,:2]
        return dict(type="RAW_TRIANGLE",face_id=int(face),triangle_ordinal=int(ordinal))

    def blocks(self, size):
        if type(size) is not int or size<1:
            raise ValueError("Positive streaming block size required")
        for start in range(0,self.count,size):
            stop=min(start+size,self.count)
            yield start,stop,self.features[start:stop]
