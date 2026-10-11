"""Lossless archival/restoration of completed, rejected derived JSON outputs."""
from pathlib import Path
import argparse,gzip,hashlib,shutil,os
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
FOLDER=ROOT/'server_rebuild/expansion_v1/physical_query_scale_full_output/control457/new_cfc732e5f068'

def main(restore=False):
 assert read(FOLDER/'receipt.json')['status']=='EXPERIMENTAL_COMPLETE_OUTPUT_SCORED_NOT_PROMOTED'
 freeze=read(FOLDER/'prediction_freeze.json')
 for branch in ('new','union'):
  original=FOLDER/f'{branch}_decoded.json';archive=original.with_suffix('.json.gz');manifest=FOLDER/f'{branch}_archive_manifest.json';expected=freeze[f'{branch}_output_sha256'];copy=Path('/tmp/weld_derived_output_cache')/expected/original.name
  if restore:
   r=read(manifest);assert digest(archive)==r['archive_sha256']
   if not copy.exists():
    copy.parent.mkdir(parents=True,exist_ok=True);partial=copy.with_suffix('.partial');assert not partial.exists()
    with gzip.open(archive,'rb') as src,partial.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
    assert digest(partial)==expected;os.replace(partial,copy)
   assert original.is_symlink() and original.readlink()==copy and digest(original)==expected
   print('RESTORED_VERIFIED',branch,flush=True);continue
  assert original.is_file() and not original.is_symlink() and not archive.exists();assert digest(original)==expected;copy.parent.mkdir(parents=True,exist_ok=False);shutil.copyfile(original,copy);assert digest(copy)==expected
  with original.open('rb') as src,gzip.open(archive,'wb',compresslevel=1) as dst:shutil.copyfileobj(src,dst,4*1024**2)
  h=hashlib.sha256()
  with gzip.open(archive,'rb') as src:
   while chunk:=src.read(4*1024**2):h.update(chunk)
  assert h.hexdigest()==expected
  r=dict(status='LOSSLESS_ARCHIVED',original_sha256=expected,archive_sha256=digest(archive),original_bytes=original.stat().st_size,archive_bytes=archive.stat().st_size,working_copy=str(copy),scope='Completed rejected derived output. Durable gzip in project; tmp may be restored. Original data/weights untouched.')
  write(manifest,r);link=original.with_suffix('.link');link.symlink_to(copy);os.replace(link,original);assert digest(original)==expected;print(branch,r['original_bytes'],r['archive_bytes'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');main(p.parse_args().restore)
