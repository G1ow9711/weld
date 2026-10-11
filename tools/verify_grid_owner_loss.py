from pathlib import Path
import torch
from weld_pipeline.server_grid_owner_loss import owner_pair_loss
from weld_pipeline.sjg_local_runtime_904 import write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 torch.manual_seed(1031);checks=0
 for n in [3,10,40]:
  for iteration in range(10):
   x=torch.randn(n,2,dtype=torch.float64,requires_grad=True);legal=torch.ones_like(x,dtype=torch.bool)
   ranges=[[[0,2],[n-2,n]],[[1,min(3,n)],[n-1,n]],[[0,2],[n-2,n]]]
   got=owner_pair_loss(x,legal,ranges);matrix=x[:,0,None]+x[None,:,1];valid=torch.ones(n,n,dtype=torch.bool).triu(1);positive=torch.zeros_like(valid)
   for (a,b),(c,d) in ranges:positive[a:b,c:d]=True
   positive &= valid
   expected=torch.logsumexp(matrix[valid],0)-torch.logsumexp(matrix[positive],0)
   assert torch.allclose(got,expected,atol=1e-12,rtol=1e-12)
   g=torch.autograd.grad(got,x,retain_graph=True)[0];h=torch.autograd.grad(expected,x)[0];assert torch.allclose(g,h,atol=1e-12,rtol=1e-12);checks+=1
 # Distinct owners: their crossed endpoints must not enter the positive union.
 x=torch.zeros(8,2,dtype=torch.float64);ranges=[[[0,1],[4,5]],[[2,3],[7,8]]]
 assert torch.allclose(owner_pair_loss(x,torch.ones_like(x,dtype=torch.bool),ranges),torch.tensor(14.).double().log())
 out=BASE/'grid_owner_loss_controls_v1';out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',dense_value_gradient_cases=checks,cross_owner_excluded=True,duplicate_owner_union=True,module_sha256=digest('weld_pipeline/server_grid_owner_loss.py')));print('PASS',checks)
if __name__=='__main__':main()
