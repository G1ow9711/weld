"""Locate permutation sensitivity without changing predictions or reading labels."""
import hashlib,json,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore'

def physical(parent):
 fields=('kind','points','endpoints_mm','length_mm','is_closed','arc_parameters','weld_type','predicted_side','material_identity','process_attributes')
 value=dict(child_count=parent['child_count'],is_closed=parent['is_closed'],gaps=parent['gaps'],segments=[{k:s[k] for k in fields if k in s} for s in parent['segments']])
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

rows=[];bags=[];unaries=[];finals=[];scores=[]
for root in (BASE,BASE/'historical_cache_order'):
 f=root/'legacy_runtime/source_10/run379/test_10';d=json.loads((f/'decoded.json').read_text());u=np.load(f/'unary.npy');keys=[physical(p) for p in d['parents']]
 bags.append(collections.Counter(keys));unaries.append(collections.Counter((k,hashlib.sha256(v.tobytes()).hexdigest()) for k,v in zip(keys,u)))
 final=json.loads((f/'final_decoded.json').read_text());finals.append(collections.Counter(physical(p) for p in final['parents']))
 with np.load(f/'precision_scores.npz') as z:scores.append(collections.Counter((k,float(p)) for k,p in zip(keys,z['probability'])))
 rows.append(dict(root=str(root),candidate_parents=len(keys),final_parents=len(final['parents'])))
report=dict(cases=rows,physical_parent_candidates_exact=bags[0]==bags[1],only_original_candidates=sum((bags[0]-bags[1]).values()),only_variant_candidates=sum((bags[1]-bags[0]).values()),physical_parent_and_unary_multisets_exact=unaries[0]==unaries[1],physical_parent_and_precision_probability_exact=scores[0]==scores[1],final_physical_parent_multisets_exact=finals[0]==finals[1],only_original_final=sum((finals[0]-finals[1]).values()),only_variant_final=sum((finals[1]-finals[0]).values()),GT_reads=0)
(BASE/'historical_cache_order/parent_permutation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
