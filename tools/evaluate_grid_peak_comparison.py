"""One fixed local-peak policy with unchanged checkpoint/query/type/pair budget."""
from pathlib import Path
import numpy as np
import torch
from tools.evaluate_grid_endpoint_comparison import run,BASE
from tools.audit_stable_pair_pool_capacity import run as capacity
from weld_pipeline.server_grid_endpoint_selector import GridEndpointSelector
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

def main():
    torch.set_num_threads(2)
    control=read(BASE/'grid_peak_candidates_controls_v1/receipt.json')
    assert control['status']=='PASS' and control['module_sha256']==digest('weld_pipeline/server_grid_peak_candidates.py')
    folder=BASE/'grid_endpoint_head_v1';receipt=read(folder/'receipt.json');checkpoint=folder/'last.pt'
    assert digest(checkpoint)==receipt['model_sha256'];state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=GridEndpointSelector().cuda().eval();model.load_state_dict(state['state_dict'])
    dest=BASE/'grid_peak_comparison_v1';dest.mkdir(exist_ok=False)
    keys=['new_cfc732e5f068','new_ceeae49d6777'];root='grid_peak_comparison_v1/peak';rows=[]
    write(dest/'protocol.json',dict(keys=keys,model_sha256=receipt['model_sha256'],policy='Local score peak plateau representatives first, fill remaining by stable score',endpoint_budget=8,pair_budget=64,spatial_thresholds=0,model_score_unchanged=True,old_event_pool_unchanged=True,baseline='grid_endpoint_comparison_v1/trained',scope='Fixed previously used development detector/query/type scope, not new independent holdout',fit_calls=0,DEV_reads=0,modules={n:digest('weld_pipeline/'+n+'.py') for n in ['server_grid_peak_candidates','server_grid_candidate_runtime']}))
    for key in keys:
        run(key,'peak',model,state,receipt['model_sha256'],output_root='grid_peak_comparison_v1',candidate_policy='local_peaks')
        capacity(key,root)
        c=read(BASE/root/key/'pair_capacity.json');score=read(BASE/root/key/'evaluation_receipt.json')['overall']
        prior=BASE/'grid_endpoint_comparison_v1/trained'/key
        with np.load(BASE/root/key/'compact_predictions.npz') as a,np.load(prior/'compact_predictions.npz') as b:
            identical=(set(a.files)==set(b.files) and all(np.array_equal(a[n],b[n],equal_nan=True) for n in a.files))
        def hits(path):return {(r['parent'],r['child']) for r in read(path)['rows'] if r['compatible_pair_count']>0}
        original=hits(BASE/'stable_pair_pool_freeze_v1'/key/'pair_capacity.json');new=hits(BASE/root/key/'pair_capacity.json');priorhits=hits(prior/'pair_capacity.json')
        rows.append(dict(key=key,G=c['reference_children'],old_grid_capacity=len(priorhits),peak_grid_capacity=len(new),original_event_capacity=len(original),original_and_peak_oracle_union=len(original|new),grid_capacity_gained=len(new-priorhits),grid_capacity_lost=len(priorhits-new),overall=score,top1_compact_arrays_identical=identical))
        print('PEAK_COMPARISON',rows[-1],flush=True)
    write(dest/'summary.json',dict(status='COMPARED_NOT_PROMOTED',rows=rows,fit_calls=0,DEV_reads=0,full_parent_assembly=False,source_sha256=digest(__file__)))
if __name__=='__main__':main()
