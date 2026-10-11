"""Bind joint-training retention scores to actual full-pool predictions."""
from pathlib import Path
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    model=BASE/'joint_arc_boundary_pointer_v1';mr=read(model/'receipt.json');assert digest(model/'model.pt')==mr['model_sha256'];first=BASE/'joint_arc_pointer_first_all_candidates_v1';second=BASE/'joint_arc_pointer_second_all_candidates_v1';fr=read(first/'receipt.json');sr=read(second/'receipt.json');assert read(first/'protocol.json')['model_sha256']==read(second/'protocol.json')['model_sha256']==mr['model_sha256'];assert digest(model/'conditional_first_family_predictions.json')==mr['first_predictions_sha256'];assert digest(model/'conditional_predictions.json')==mr['predictions_sha256'];conditional_first=read(model/'conditional_first_family_predictions.json')['rows'];conditional_second=read(model/'conditional_predictions.json')['rows'];bank={}
    for face in fr['faces']:
        p=first/face['file'];assert digest(p)==face['sha256'];bank[face['face']]=read(p)['rows']
    for row in conditional_first:assert bank[row['face']][row['candidate']]['selected_local_event_indices']==row['selected_local_event_indices']
    assert digest(second/'predictions.json')==sr['predictions_sha256'];rows=read(second/'predictions.json')['rows']
    for row in conditional_second:assert rows[row['profile']]['selected_local_event_indices']==row['selected_local_event_indices']
    scores={}
    for name,folder in [('first_before_second_training',BASE/'arc_pointer_all_candidates_v1'),('first_after_second_only',BASE/'arc_pointer_first_family_retention_v1'),('first_joint',first),('second_before_joint',BASE/'curved_group_comparison_grouped_all_candidates_v1'),('second_joint',second)]:
        path=folder/'geometry_diagnostic.json';r=read(path);scores[name]=dict(score=r['score']['categories']['overall'],score_sha256=digest(path))
    out=BASE/'joint_arc_retention_audit_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(scores=scores,first_conditional_full_pool_parity=len(conditional_first),second_conditional_full_pool_parity=len(conditional_second),model_sha256=mr['model_sha256'],scope='Two previously trained families, different input branches and explicit fixed-H/parent diagnostic conventions retained. Retention, not independent generalization or contract acceptance. No automatic promotion.',DEV_reads=0));print(scores,flush=True)

if __name__=='__main__':main()
