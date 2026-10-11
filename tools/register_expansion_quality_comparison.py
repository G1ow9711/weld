"""Register a fixed-budget structure coverage comparison before expansion fits."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cohort=BASE/'coverage_cohort_protocol.json';cases=json.loads(cohort.read_text())['cases']
protocol=dict(status='REGISTERED_BEFORE_EXPANSION_QUALITY_FIT',experiment='More TRAIN object families at equal known-candidate budget',seed=1031,
 old_keys=['21','BM79A_native_local','10','25','36'],expansion_keys=[r['key'] for r in cases],
 input_protocol_sha256=sha(cohort),group_manifest_sha256=sha(BASE/'group_manifest.json'),
 feature_contract='SIDE_SYMMETRIC_REAL_CARRIER_CHILD466_PARENT1458_V1',
 candidate_policy='Freeze actual XYZ-derived hypotheses and original composite defaults before reading candidate supervision; do not inject GT geometry or prune by GT.',
 supervision=dict(eligible='Every child has a valid original D3 geometry correspondence to an annotated TRAIN child; retain original whole-parent D3 TP/FP counts, including structural incompatibility among known children.',unknown='Mask parent loss if any child lacks valid correspondence.',support_tube_expansion=False,negative_from_absence=False,annotation_closure_claimed=False),
 folds=dict(held_out='Each original key in turn, exclude its full conservative family from both arms. Expansion-family duplicate exclusions follow frozen group_manifest.',A='Original TRAIN families other than heldout',B='Same original families plus registered expansion TRAIN families'),
 budget=dict(total_rows='N = min(number of eligible unique candidates in A, number in B); determine from eligibility only before either fit.',sampling='Within each family sort by SHA256(seed || family_key || parent canonical signature); select without replacement using round-robin across sorted family keys until N rows. Deduplicate canonical physical parent signatures within family before sampling; do not merge distinct contact references.',weights='Equal per selected candidate in both arms.',freeze='Save exact row indexes, canonical signatures, features/parents/labels source hashes, exclusions and N before model.fit.'),
 estimator=dict(kind='ExtraTreesRegressor',n_estimators=96,max_depth=18,min_samples_leaf=2,max_features=.75,random_state=1031,n_jobs=2),
 targets='[D3_TP / child_count, D3_FP / child_count]; multiply predictions by child_count for selection utility TP-FP, lambda=1.',
 isolate='Use the same existing heldout competition_v4 predictions, the same candidate resource constraints and canonical selector in A and B. Competition is not refit for this diagnostic.',
 outcomes=['Per-heldout D3 TP/FP overall and by Hline/Vline/HArc','Known-row count MAE','Unseen versus seen parent-structure strata registered from TRAIN labels only','Unknown candidate positive-utility count'],
 interpretation='Conditional quality-head coverage diagnostic only: upstream models are frozen shared models. Not full-pipeline out-of-fold, independent validation, or formal acceptance.',
 protected=['Original DEV64 family','15 EXPANSION_DEV_VALID families','6 quarantined TRAIN label conflicts','Legacy families with unresolved role'],
 do_not_tune='No outcome-based threshold, tree, candidate-budget, or family-selection sweep. Failed coverage remains a reported result.',fit_calls=0,source_sha256=sha(Path(__file__)))
out=BASE/'quality_comparison_protocol.json';assert not out.exists();out.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n');print(out)
