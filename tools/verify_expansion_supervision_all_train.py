"""Compare new generic strict supervision with every original TRAIN row."""
from pathlib import Path
from threadpoolctl import threadpool_limits
import tools.prepare_expansion_parent_supervision as m
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
DEST=m.EXP/'routing_controls/supervision_all_train'
def main():
 DEST.mkdir(exist_ok=False);m.BASE=DEST;m.LABELS=m.ROOT/'server_rebuild/labels_v1';reports=[]
 fields=['hypothesis','children','D3_TP','D3_FP','known_geometry_children','parent_loss_valid']
 for key in ['21','BM79A_native_local','10','25','36']:
  src=m.ROOT/'server_rebuild/unified_parent_scoring_v1'/('source_'+key);dest=DEST/('source_'+key);dest.mkdir();r=read(src/'receipt.json')
  for name in ['receipt.json',*r['files']]:(dest/name).symlink_to(src/name)
  m.run(key);actual=read(dest/'d3_supervision.json');old=read(src/'d3_supervision.json');assert len(actual['rows'])==len(old['rows'])
  for a,b in zip(actual['rows'],old['rows']):
   for k in fields:assert a[k]==b[k],(key,k,a,b)
  row=dict(key=key,rows=len(actual['rows']),fields=fields,actual_sha256=digest(dest/'d3_supervision.json'),reference_sha256=digest(src/'d3_supervision.json'),exact_match=True);reports.append(row);write(DEST/'progress.json',dict(completed=reports));print('EXACT_SUPERVISION',key,len(actual['rows']),flush=True)
 write(DEST/'receipt.json',dict(status='PASS',scope='All original TRAIN parents; strict D3 supervision values match, including arcs/closed and multi-child structures. No expansion quality claim.',cases=reports,source_sha256=digest(m.__file__),fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
