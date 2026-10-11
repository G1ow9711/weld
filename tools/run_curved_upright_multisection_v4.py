"""Input-determined multi-height section coverage, still experimental carriers."""
import argparse,time
from tools.probe_xyz_curved_upright_sections_v4 import run,BASE
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from threadpoolctl import threadpool_limits
p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();start=time.perf_counter();dest=BASE/'curved_upright_probe_v4'/a.key
with threadpool_limits(limits=2):
 run(a.key,0);plan=read(dest/'section_000/section_plan.json');profiles=[];bindings=[]
 for i in range(len(plan['levels_mm'])):
  if i:run(a.key,i)
  folder=dest/('section_%03d'%i);r=read(folder/'receipt.json');new=read(folder/'profiles.json')
  for row in new:row['source_section_index']=i
  profiles.extend(new);bindings.append(dict(index=i,receipt_sha256=digest(folder/'receipt.json'),profiles=len(new)));print('SECTION_COMPLETE',a.key,i,len(plan['levels_mm']),len(new),flush=True)
 write(dest/'profiles.json',profiles);write(dest/'receipt.json',dict(status='XYZ_MULTI_SECTION_CARRIERS_NOT_DEPLOYED',key=a.key,profiles=len(profiles),sections=bindings,input_sha256=plan['input_sha256'],profiles_sha256=digest(dest/'profiles.json'),GT_reads=0,STEP_reads=0,fit_calls=0,finite_contact_proven=False,chain_gap_is_not_weld_segmentation_or_35mm_rule=True,seconds=time.perf_counter()-start))
