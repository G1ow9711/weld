"""Write fresh pure-PCD interfaces for offline training, never read them here."""
import json,hashlib
from pathlib import Path
import numpy as np

def json_value(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [json_value(v) for v in value]
 return value

class FreshTrainingCapture:
 def __init__(self,directory):
  self.directory=Path(directory);self.directory.mkdir(exist_ok=False);self.files={};self.sequence_calls=0
 def array_stream(self,name,shape,dtype,blocks):
  path=self.directory/name;dtype=np.dtype(dtype);h=hashlib.sha256();rows=0
  full_hash=hashlib.sha256()
  with path.open('xb') as f:
   class HashWriter:
    def write(self,value):full_hash.update(value);return f.write(value)
    def __getattr__(self,name):return getattr(f,name)
   writer=HashWriter()
   np.lib.format.write_array_header_2_0(writer,dict(descr=np.lib.format.dtype_to_descr(dtype),fortran_order=False,shape=tuple(shape)))
   for block in blocks:
    a=np.asarray(block,dtype=dtype);assert a.shape[1:]==tuple(shape[1:]);rows+=len(a)
    raw=a.tobytes(order='C');h.update(raw);writer.write(raw)
  assert rows==shape[0]
  self.files[name]=dict(shape=list(shape),dtype=str(dtype),array_payload_sha256=h.hexdigest(),
    file_sha256=full_hash.hexdigest(),file_bytes=path.stat().st_size)
 def original_xyz_mm(self,xyz):
  self.array_stream('original_xyz_mm.npy',xyz.shape,xyz.dtype,
    (xyz[s:s+65536]*1000 for s in range(0,len(xyz),65536)))
 def sequences(self,sequences,geometry):
  self.sequence_calls+=1;assert self.sequence_calls==1 and sequences
  metadata=[];first=0
  for seq in sequences:
   n=len(seq['xyz']);assert len(seq['features'])==n
   metadata.append(dict(start=first,stop=first+n,
    original_xyz_dtype=str(seq['xyz'].dtype),original_features_dtype=str(seq['features'].dtype),
    fields=json_value({k:v for k,v in seq.items() if k not in ['xyz','features']})))
   first+=n
  for key in ['xyz','features']:
   dtype=np.result_type(*[s[key].dtype for s in sequences]);cols=sequences[0][key].shape[1]
   assert all(s[key].ndim==2 and s[key].shape[1]==cols for s in sequences)
   self.array_stream(key+'.npy',(first,cols),dtype,
    (s[key][i:i+65536] for s in sequences for i in range(0,len(s[key]),65536)))
  self.metadata=dict(sequences=metadata,geometry=json_value(geometry),total_rows=first,
    original_factory_row_order_and_values_preserved=True,GT_reads=0,STEP_reads=0,
    capture_files_are_training_outputs_never_runtime_prediction_inputs=True)
 def probabilities(self,sequences,probabilities):
  assert self.sequence_calls==1 and len(sequences)==len(probabilities)
  assert all(len(s['xyz'])==len(p) and p.shape[1]==7 for s,p in zip(sequences,probabilities))
  n=sum(len(p) for p in probabilities);dtype=np.result_type(*[p.dtype for p in probabilities])
  self.array_stream('original_parent_probabilities.npy',(n,7),dtype,
   (p[i:i+65536] for p in probabilities for i in range(0,len(p),65536)))
 def structure(self,structural,process):
  for name,a in [('structural.npy',structural),('process.npy',process)]:
   self.array_stream(name,a.shape,a.dtype,(a[i:i+65536] for i in range(0,len(a),65536)))
 def finish(self):
  assert self.sequence_calls==1
  self.metadata['files']=self.files
  self.metadata['scope']='Fresh original raw factories and frozen parent outputs; offline training interfaces only, no GT attached.'
  with (self.directory/'manifest.json').open('x',encoding='utf-8') as f:
   json.dump(self.metadata,f,ensure_ascii=False,indent=2,allow_nan=False)
  return self.metadata
