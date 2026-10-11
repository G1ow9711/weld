"""Hash-bound curve metadata alongside the existing read-only array storage."""
import json
from pathlib import Path
from training.labels import sha256
from weld_pipeline.mapped_cache import write_piece as write_base,load_piece as load_base

STORAGE_FORMAT='curved_sequence_memmap_v1'


def write_piece(folder,piece,provenance):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    metadata={'storage_format':STORAGE_FORMAT,'sequences':[
        {k:s[k] for k in ['curve_kind','curve','candidate_id'] if k in s} for s in piece['sequences']]}
    text=json.dumps(metadata,ensure_ascii=False,sort_keys=True,indent=2)
    path=folder/'curve_metadata.json'
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('Curve metadata changed; use a new dataset version')
    path.write_text(text,encoding='utf-8')
    return write_base(folder,piece,{**provenance,'curve_metadata_sha256':sha256(path)})


def load_piece(folder,verify=True):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    path=folder/'curve_metadata.json'
    if sha256(path)!=manifest['provenance']['curve_metadata_sha256']:raise ValueError('Curve metadata hash differs')
    metadata=json.loads(path.read_text(encoding='utf-8'))
    if metadata['storage_format']!=STORAGE_FORMAT:raise ValueError('Unknown curve metadata format')
    piece=load_base(folder,verify=verify)
    if len(metadata['sequences'])!=len(piece['sequences']):raise ValueError('Curve metadata count differs')
    for seq,extra in zip(piece['sequences'],metadata['sequences']):
        if set(extra)-{'curve_kind','curve','candidate_id'}:raise ValueError('Unexpected curve metadata key')
        seq.update(extra)
    return piece
