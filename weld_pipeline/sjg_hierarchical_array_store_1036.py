"""Lossless NPZ-to-NPY conversion and per-source read-only mapping."""
import argparse,gc,hashlib,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources

def array_digest(a):
    assert a.flags.c_contiguous
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

class ArrayStore:
    def __init__(self,path):
        self.path=path;self.bindings=read(path/'arrays.json')['arrays'];self.files=list(self.bindings);self.loaded={}
    def __enter__(self):return self
    def __getitem__(self,key):
        if key not in self.loaded:self.loaded[key]=np.load(self.path/self.bindings[key]['file'],mmap_mode='r',allow_pickle=False)
        return self.loaded[key]
    def __exit__(self,*args):
        for a in self.loaded.values():a._mmap.close()
        self.loaded.clear()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False);start=time.perf_counter();data=OUT/'run206';manifest=read(data/'training_manifest.json')
    static={b['source_key']:b for b in manifest['static_bindings']};items=[]
    for case in manifest['cases']:
        for kind,b in [('static',static[case['source_key']]),('train',case)]:
            check_resources();source=data/b['file'];assert digest(source)==b['sha256']
            folder=dest/(case['source_key']+'_'+kind);folder.mkdir();arrays={}
            with np.load(source,allow_pickle=False) as z:
                for key in z.files:
                    value=z[key];assert value.flags.c_contiguous
                    path=folder/(key+'.npy');np.save(path,value,allow_pickle=False)
                    mapped=np.load(path,mmap_mode='r',allow_pickle=False)
                    original_digest=array_digest(value)
                    assert value.shape==mapped.shape and value.dtype==mapped.dtype and original_digest==array_digest(mapped)
                    arrays[key]=dict(file=path.name,shape=list(value.shape),dtype=str(value.dtype),order='C',
                        content_sha256=original_digest,file_sha256=digest(path),bytes=int(value.nbytes))
                    mapped._mmap.close();del value,mapped;gc.collect()
            write(folder/'arrays.json',dict(source_npz=str(source),source_sha256=b['sha256'],arrays=arrays))
            items.append(dict(source=case['source_key'],kind=kind,folder=folder.name,manifest_sha256=digest(folder/'arrays.json')))
            print('ARRAY_MAPPING_VERIFIED',case['source_key'],kind,sum(v['bytes'] for v in arrays.values()),flush=True)
    write(dest/'receipt.json',dict(status='LOSSLESS_ARRAY_STORE_READY',items=items,data_manifest_sha256=digest(data/'training_manifest.json'),
        converted_seconds=time.perf_counter()-start,optimizer_updates=0,TRAIN_G=363,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
