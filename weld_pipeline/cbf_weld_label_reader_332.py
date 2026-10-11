"""Offline-only extraction of original weld JSON from BinOcaf AsciiString.

No CBF contents are executed. No pose axis is interpreted as a plate normal.
The exact original JSON, ObjectIds, reference links and field values survive.
This reader must never become an inference input.
"""
from pathlib import Path
import hashlib
import json
import math
import struct
import uuid

VERSION = "CUSTOMER_BIN_OCAF_ASCII_WELD_JSON_LABEL_READER_332_v1"


def extract_cbf_weld_json(path):
    path = Path(path)
    data = path.read_bytes()
    if not 352 <= len(data) <= 16 * 1024 * 1024:
        raise ValueError("Unsupported label-document size")
    if not data.startswith(b"BINFILE") or b"FILE_FORMAT: BinOcaf" not in data[:512]:
        raise ValueError("Not the customer BinOcaf document format")
    if b"TDataStd_AsciiString" not in data[:512]:
        raise ValueError("No supported AsciiString attribute in the type table")
    candidates, offset = [], 0
    while True:
        offset = data.find(b"[", offset)
        if offset < 0:
            break
        start = offset
        offset += 1
        if start < 4:
            continue
        recorded_size = struct.unpack_from("<I", data, start - 4)[0]
        if not 32 <= recorded_size <= len(data) - start:
            continue
        end = data.find(b"\0", start, start + recorded_size)
        if end < 0 or recorded_size != (end - start) + 1 + 16:
            continue
        payload = data[start:end]
        for encoding in ("utf-8", "cp936"):
            try:
                text = payload.decode(encoding)
                if text.encode(encoding) != payload:
                    continue
                objects = json.loads(text)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(objects, list) or not objects:
                continue
            if not all(isinstance(o, dict) and "ObjectId" in o and "MatrixCollection" in o for o in objects):
                continue
            ids = [str(o["ObjectId"]) for o in objects]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate original parent ObjectId")
            for object_id in ids:
                uuid.UUID(object_id)
            candidates.append((objects, dict(
                version=VERSION, original_cbf_path=str(path),
                original_cbf_sha256=hashlib.sha256(data).hexdigest(),
                original_cbf_bytes=len(data), json_payload_byte_offset=start,
                json_payload_bytes=len(payload), attribute_payload_recorded_bytes=recorded_size,
                original_payload_sha256=hashlib.sha256(payload).hexdigest(),
                source_encoding=encoding, encoding_roundtrip_exact=True,
                ascii_attribute_guid=str(uuid.UUID(bytes_le=data[end + 1:end + 17])),
                original_parent_objects=len(objects),
                original_strings_values_and_references_retained=True,
                runtime_input_permitted=False)))
            break
    if len(candidates) != 1:
        raise ValueError("Expected one unambiguous framed original weld JSON attribute")
    return candidates[0]


class JsonReferenceIndex:
    """Resolve requested Json.NET references without expanding cyclic trees."""

    def __init__(self, objects):
        self.nodes = {}
        pending = [objects]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                if "$id" in node:
                    key = str(node["$id"])
                    if key in self.nodes:
                        raise ValueError("Duplicate serialized reference identifier")
                    self.nodes[key] = node
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)

    def resolve(self, node):
        seen = set()
        while isinstance(node, dict) and "$ref" in node:
            if set(node) != {"$ref"}:
                raise ValueError("Ambiguous mixed reference and data")
            key = str(node["$ref"])
            if key in seen or key not in self.nodes:
                raise ValueError("Cyclic or absent requested reference")
            seen.add(key)
            node = self.nodes[key]
        return node

    def xyz(self, node):
        node = self.resolve(node)
        point = [float(node[key]) for key in ("X", "Y", "Z")]
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Nonfinite original vector")
        return point

    def translation(self, matrix):
        matrix = self.resolve(matrix)
        if matrix.get("IsNaN", False):
            raise ValueError("Original pose translation is NaN")
        point = self.resolve(matrix["T"])
        if abs(float(point.get("W", 1.0)) - 1.0) > 1e-12:
            raise ValueError("Pose translation has a non-point homogeneous coordinate")
        return self.xyz(point)

    def line_children(self, parent):
        collection = self.resolve(parent["MatrixCollection"])
        if not isinstance(collection, list):
            raise ValueError("Original MatrixCollection is not a list")
        children = []
        for index, native in enumerate(collection):
            native = self.resolve(native)
            # A stored MidMatrix alone is not evidence of a curved child.
            if native["HasMid"] or native.get("ChildMatrixModels") is not None:
                raise ValueError("Curved or nested native geometry requires its own reader")
            start, end = self.translation(native["StartMatrix"]), self.translation(native["EndMatrix"])
            length = math.dist(start, end)
            if length <= 0:
                raise ValueError("Original native line has coincident endpoints")
            children.append(dict(original_cbf_child_index=index,
                original_serialized_child_id=native.get("$id"),
                original_kind="line", points_reference_mm=[start, end],
                length_mm=length, original_is_vertical=bool(native["isVerticalWeldLine"]),
                raw_customer_matrix_model=native))
        return children
