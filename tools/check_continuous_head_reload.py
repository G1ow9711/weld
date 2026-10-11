import hashlib,json,pathlib,numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];p=ROOT/'server_rebuild/continuous_head_v1';r=json.loads((p/'receipt.json').read_text())
for name,digest in r['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
torch.set_num_threads(2);ck=torch.load(p/'model.pt',weights_only=True,map_location='cpu');m=torch.nn.Sequential(torch.nn.Linear(ck['input_dim'],128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,10)).eval();m.load_state_dict(ck['state_dict']);norm=np.load(p/'normalization.npz');original=np.load(p/'TRAIN_query_outputs.npy',mmap_mode='r');cursor=0;maxerror=0.;rows=0
for key in r['bindings']:
 x=np.load(p/(key+'.npz'))['features'];ids=np.linspace(0,len(x)-1,min(1000,len(x)),dtype=int)
 with torch.no_grad():out=m(torch.from_numpy((x[ids]-norm['mean'])/norm['scale'])).numpy()
 maxerror=max(maxerror,float(abs(out-original[cursor+ids]).max()));cursor+=len(x);rows+=len(ids)
assert maxerror<1e-3
result=dict(status='CONTINUOUS_HEAD_RELOAD_PASS',rows=rows,max_cpu_gpu_raw_output_difference=maxerror,checkpoint_sha256=r['files']['model.pt'],full_accuracy_test=False)
(p/'reload_control.json').write_text(json.dumps(result,indent=2));print(result)
