"""Fixed-H independent-interval geometry diagnostic, NOT learned semantics."""
from pathlib import Path
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc,CurveGeometryError
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.contract_geometry_d3_409 import score_source
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main(model_name='curved_probe_boundary_pointer_v1', output_name='curved_probe_grouped_pointer_all_candidates_v1'):
    key='new_fcaf2af1abee';folder=BASE/output_name;receipt=read(folder/'receipt.json');assert digest(folder/'predictions.json')==receipt['predictions_sha256'];events=BASE/'curved_probe_event_groups_v1'/key;ep=read(BASE/'curved_probe_vertex_events_v1'/key/'protocol.json');source=BASE/'curved_upright_probe_v3'/key;assert digest(source/'profiles.json')==ep['profiles_sha256'];profiles=read(source/'profiles.json');assert digest(events/'events.npz')==read(events/'receipt.json')['files']['events.npz'];predictions=read(folder/'predictions.json')['rows'];parents=[];invalid=[]
    with np.load(events/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    for row in predictions:
        index=row['profile'];a,b=offsets[index:index+2];q=positions[a:b];path=row['selected_local_event_indices'];arc=CircularArc.from_parameters(profiles[index]['representation']['arc_parameters']);assert all(0<=i<len(q) for i in path);assert all(q[j]>q[i] for i,j in zip(path,path[1:]))
        if len(path)==1:invalid.append(dict(counted_false_children=1,predicted_weld_types=[0],predicted_kinds=[2],reason='ONE_BOUNDARY'))
        for i,j in zip(path,path[1:]):
            try:
                segment=sample_arc(arc.subarc(float(q[i]),float(q[j])));segment['weld_type']='horizontal';reverse_segment(segment);parents.append(dict(child_count=1,is_closed=False,gaps=[],segments=[segment]))
            except CurveGeometryError as error:invalid.append(dict(counted_false_children=1,predicted_weld_types=[0],predicted_kinds=[2],reason=str(error)))
    model=BASE/model_name;mr=read(model/'receipt.json');assert digest(model/'conditional_predictions.json')==mr['predictions_sha256'];conditional=read(model/'conditional_predictions.json')['rows'];parity_expected=read(model/'protocol.json').get('mode')=='grouped'
    if parity_expected:assert all(predictions[r['profile']]['selected_local_event_indices']==r['selected_local_event_indices'] for r in conditional)
    target=BASE/'labels'/key/'targets.json';truth=native_targets(read(target));score=score_source(dict(parents=parents,invalid_predictions=invalid,total_prediction_children=len(parents)+sum(x['counted_false_children'] for x in invalid)),truth,[False]*len(truth['parents']));write(folder/'geometry_diagnostic.json',dict(score=score,invalid_predictions=invalid,conditional_parity_count=len(conditional) if parity_expected else None,target_sha256=digest(target),prediction_receipt_sha256=digest(folder/'receipt.json'),scope='Every adjacent predicted boundary interval is emitted as independent parent with fixed horizontal type. Parent identity, gap suppression, weld existence/contact and type NOT learned; all duplicates and gap emissions count. Zero-fit grouped-feature intervention, TRAIN only, not end-to-end acceptance.',DEV_reads=0,no_automatic_promotion=True));print(score['categories']['overall'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model-name',default='curved_probe_boundary_pointer_v1');p.add_argument('--output-name',default='curved_probe_grouped_pointer_all_candidates_v1');a=p.parse_args();main(a.model_name,a.output_name)
