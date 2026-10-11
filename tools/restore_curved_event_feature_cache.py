"""Rebuild a temporary working copy from the verified project-local gzip."""
from pathlib import Path
import gzip,shutil,os
from weld_pipeline.sjg_local_runtime_904 import read,digest
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'server_rebuild/expansion_v1/curved_probe_event_features_v1/new_fcaf2af1abee';manifest=read(source/'archive_manifest.json');archive=source/'features.npy.gz';assert digest(archive)==manifest['archive_sha256'];copy=Path(manifest['working_copy']);assert copy.is_relative_to('/tmp/weld_derived_feature_cache')
    if not copy.exists():
        copy.parent.mkdir(parents=True,exist_ok=True);partial=copy.with_suffix('.partial');assert not partial.exists()
        with gzip.open(archive,'rb') as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
        assert digest(partial)==manifest['original_sha256'];os.replace(partial,copy)
    assert digest(copy)==manifest['original_sha256'];original=source/'features.npy';assert original.is_symlink() and original.readlink()==copy;assert digest(original)==manifest['original_sha256'];print('Verified working cache available',flush=True)

if __name__=='__main__':main()
