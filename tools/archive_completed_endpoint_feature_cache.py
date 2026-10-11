"""Losslessly archive a completed derived cache; preserve transparent live reads."""
from pathlib import Path
import gzip,hashlib,shutil,os,json
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'server_rebuild/expansion_v1/frozen_features/legacy_runtime/source_new_ac2008a03607/run421/endpoints/source_new_ac2008a03607'


def main():
    original=SOURCE/'features.npy';receipt=read(SOURCE/'receipt.json');expected=receipt['feature_sha256'];assert original.is_file() and not original.is_symlink();assert digest(original)==expected
    working=Path('/tmp/weld_derived_feature_cache')/expected;working.mkdir(parents=True,exist_ok=False);copy=working/'features.npy';shutil.copyfile(original,copy);assert digest(copy)==expected
    archive=SOURCE/'features.npy.gz';assert not archive.exists()
    with original.open('rb') as src,gzip.open(archive,'wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst,4*1024**2)
    checksum=hashlib.sha256()
    with gzip.open(archive,'rb') as stream:
        while chunk:=stream.read(4*1024**2):checksum.update(chunk)
    assert checksum.hexdigest()==expected
    manifest=dict(status='LOSSLESS_DERIVED_CACHE_ARCHIVE_WITH_TEMP_WORKING_LINK',original_sha256=expected,original_bytes=original.stat().st_size,archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,working_copy=str(copy),scope='Only completed derived endpoint-feature cache. Original checkpoint/data untouched. gzip in project is durable; /tmp copy may be rebuilt from gzip after restart. Existing canonical features.npy reads preserved via symlink.')
    write(SOURCE/'archive_manifest.json',manifest);link=SOURCE/'features.npy.link';link.symlink_to(copy);os.replace(link,original);assert digest(original)==expected;print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
