"""Convert frozen pilot files one piece at a time, preserving all array values."""
import argparse
import gc
import json
import shutil
from pathlib import Path
import torch
from training.labels import sha256
from weld_pipeline.mapped_cache import STORAGE_FORMAT,write_piece


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='weld_pipeline/data/surface_pilot_v1')
    ap.add_argument('--outdir',default='weld_pipeline/data/surface_mapped_v1')
    ap.add_argument('--groups',nargs='+')
    args=ap.parse_args();source,out=Path(args.source),Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    split=json.loads((source/'split.json').read_text(encoding='utf-8'))
    split.update(storage_format=STORAGE_FORMAT,parent_split_sha256=sha256(source/'split.json'))
    split_path=out/'split.json'
    if split_path.exists() and json.loads(split_path.read_text(encoding='utf-8'))!=split:raise ValueError('Frozen split differs')
    split_path.write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    snapshot=out/'conversion_source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__),Path('weld_pipeline/mapped_cache.py')]:
        dest=snapshot/p.name
        if dest.exists() and sha256(dest)!=sha256(p):raise ValueError('Conversion source differs')
        if not dest.exists():shutil.copy2(p,dest)
    for name in args.groups or split['train']+split['validation']:
        folder=out/name
        if not folder.resolve().is_relative_to(out.resolve()):raise ValueError('Invalid group name')
        path=source/(name+'.pt')
        provenance={'parent_cache':str(path),'parent_sha256':sha256(path)}
        if (folder/'manifest.json').exists():
            from weld_pipeline.mapped_cache import load_piece
            piece=load_piece(folder,verify=True)
            if json.loads((folder/'manifest.json').read_text(encoding='utf-8'))['provenance']!=provenance:raise ValueError('Parent differs')
            print('VERIFIED_EXISTING',name,flush=True)
        else:
            with path.open('rb') as f:piece=torch.load(f,map_location='cpu')
            manifest=write_piece(folder,piece,provenance)
            print('MAPPED',name,'sequences',len(manifest['sequences']),'points',manifest['files']['features.npy']['shape'][0],flush=True)
        del piece;gc.collect()
    if all((out/n/'manifest.json').exists() for n in split['train']+split['validation']):
        (out/'conversion_complete.json').write_text(json.dumps({'groups':len(split['train'])+len(split['validation']),
            'storage_format':STORAGE_FORMAT,'content_preserved':True},indent=2),encoding='utf-8')


if __name__=='__main__':main()
