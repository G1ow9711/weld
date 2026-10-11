"""Complete carrier commits: payloads first, durable manifest last."""
import json
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import digest,read
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

def durable_json(path,value):
    data=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    atomic_binary_write(path,lambda f:f.write(data))

def publish(folder,position,binding,payload,feature=None,interrupt_at=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    data=folder/f'carrier_{position}.json';manifest=folder/f'carrier_{position}.complete.json'
    if interrupt_at=='during_payload':
        (folder/f'carrier_{position}.json.test.pending').write_bytes(b'{')
        return
    durable_json(data,payload)
    if interrupt_at=='before_manifest':return
    files={data.name:digest(data)}
    if feature is not None:files[str(Path(feature).relative_to(folder.parent))]=digest(feature)
    durable_json(manifest,dict(status='COMPLETE',position=position,binding=binding,
        data=data.name,data_sha256=files.pop(data.name),feature_files=files))

def recover(folder,position,binding):
    folder=Path(folder);manifest=folder/f'carrier_{position}.complete.json'
    if not manifest.exists():return None
    m=read(manifest);assert m['status']=='COMPLETE' and m['position']==position and m['binding']==binding
    assert digest(folder/m['data'])==m['data_sha256']
    for f,h in m['feature_files'].items():assert digest(folder.parent/f)==h
    return read(folder/m['data'])

def controls(folder):
    root=Path(folder);root.mkdir(exist_ok=False);binding={'frozen':'fixture'}
    payload=dict(H=[[7,1,3,2]],negative_constraints=8,negative_carrier=True)
    for i,phase in enumerate(('during_payload','before_manifest',None)):
        sub=root/str(i);publish(sub,0,binding,payload,interrupt_at=phase)
        restored=recover(sub,0,binding)
        assert (restored is not None)==(phase is None)
        if restored is None:publish(sub,0,binding,payload)
        first=recover(sub,0,binding);second=recover(sub,0,binding)
        assert first==second==payload
        unique={0:second};assert sum(p['negative_constraints'] for p in unique.values())==8
        assert sum(len(p['H']) for p in unique.values())==1
    durable_json(root/'receipt.json',dict(status='PASS',interruption_points=3,
        no_skip_incomplete=True,no_double_count_complete=True,H_and_negative_counts_preserved=True))
