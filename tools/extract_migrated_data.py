"""Extract the approved data archive without overwriting data; verify CRC and SHA."""
import hashlib,json,os,pathlib,stat,time,zipfile
root=pathlib.Path('/root/autodl-tmp/weld_cloud/data/G_Data')
archive=root/'129.zip'
report=pathlib.Path(__file__).resolve().parents[1]/'migration_checks'
report.mkdir(exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''): h.update(b)
 return h.hexdigest()
manifest=json.loads((root/'_source_manifest.json').read_text())
expected=next(x for x in manifest['files'] if x['relative_path']=='129.zip')
sha=digest(archive)
assert archive.stat().st_size==expected['bytes'] and sha==expected['sha256']
records=[]; start=time.time()
with zipfile.ZipFile(archive) as z:
 names=set()
 for info in z.infolist():
  name=info.filename
  if not info.flag_bits & 0x800:
   try:name=name.encode('cp437').decode('gb18030')
   except (UnicodeError,LookupError):pass
  rel=pathlib.PurePosixPath(name)
  assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in name
  assert not stat.S_ISLNK(info.external_attr>>16)
  assert name not in names;names.add(name)
  target=root/rel
  assert target.resolve().is_relative_to(root.resolve())
  if info.is_dir():target.mkdir(parents=True,exist_ok=True);continue
  target.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256();size=0
  if target.exists():
   with z.open(info) as f:
    for b in iter(lambda:f.read(4194304),b''):h.update(b);size+=len(b)
   assert size==target.stat().st_size and h.hexdigest()==digest(target),str(target)
  else:
   temp=target.with_name(target.name+'.extracting')
   with z.open(info) as f,temp.open('xb') as out:
    for b in iter(lambda:f.read(4194304),b''):out.write(b);h.update(b);size+=len(b)
   assert size==info.file_size
   os.replace(temp,target)
  records.append(dict(path=str(rel),bytes=size,sha256=h.hexdigest(),zip_crc32=info.CRC))
  if len(records)%100==0:print(json.dumps(dict(files=len(records),bytes=sum(x['bytes'] for x in records),seconds=time.time()-start)),flush=True)
result=dict(status='EXTRACTED_CRC_AND_CONTENT_HASHED',archive_sha256=sha,files=records,seconds=time.time()-start)
(report/'129_extraction_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(dict(status=result['status'],files=len(records),seconds=result['seconds'])),flush=True)
