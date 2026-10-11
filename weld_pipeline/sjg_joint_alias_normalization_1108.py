"""Input-only canonicalization of proven duplicate plane-pair generations.

941.envelope_profiles recomputes the same ordered plane intersection and clips
its finite domain. If full finite extent, ordered observed faces, side/material
hypothesis and every support-state port agree to floating-point roundoff, this
is a second generation of the SAME input object. No material truth is inferred.
Different faces, side hypotheses, domains or relation interpretations survive.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_object_features_1106 import side_rows

KEYS=('21','4','10','25','36','42','BM79A_native_local')
PROVEN_BRANCHES={'global_observed_plane_pair','finite_convex_envelope'}


def normalize(profiles,offsets,spans):
    representative=np.arange(len(spans),dtype=np.int64);profile_rep=np.arange(len(profiles),dtype=np.int64)
    groups={};merges=[]
    for pid,p in enumerate(profiles):
        if p['representation']['kind']!='line' or p.get('proposal_branch') not in PROVEN_BRANCHES:continue
        faces=p.get('face_indices',[])
        if len(faces)!=2:continue
        key=json.dumps([faces,p.get('material_pair'),p.get('weld_side'),p.get('side_hypothesis'),
                        p.get('weld_type')],sort_keys=True,separators=(',',':'))
        endpoints=np.asarray(p['representation']['endpoints_mm'],np.float64)
        lo,hi=offsets[pid:pid+2];rows=spans[lo:hi]
        matched=None
        for candidate in groups.get(key,[]):
            reference=profiles[candidate];rlo,rhi=offsets[candidate:candidate+2];other=spans[rlo:rhi]
            if rows.shape!=other.shape or not np.array_equal(rows[:,2:],other[:,2:]):continue
            ref=np.asarray(reference['representation']['endpoints_mm'],np.float64)
            scale=max(1.,float(np.abs(endpoints).max()),float(np.abs(ref).max()),
                      float(rows[-1,1]),float(other[-1,1]))
            tolerance=64*np.finfo(float).eps*scale
            error=max(float(np.max(np.abs(endpoints-ref))),float(np.max(np.abs(rows[:,:2]-other[:,:2]))))
            if error<=tolerance:
                matched=candidate;representative[lo:hi]=np.arange(rlo,rhi)
                profile_rep[pid]=candidate
                merges.append(dict(profile=pid,representative_profile=candidate,
                    ordered_face_indices=faces,fragments=int(hi-lo),maximum_roundoff_difference_mm=error,
                    numeric_roundoff_bound_mm=tolerance,
                    generator_branches=[reference.get('proposal_branch'),p.get('proposal_branch')]))
                break
        if matched is None:groups.setdefault(key,[]).append(pid)
    return representative,profile_rep,merges


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run288');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);rows=[]
    freeze=read(OUT/'run285/all_geometry_frozen_before_labels.json')
    # Freeze all input equivalence maps before loading any witness responsibility.
    for key in KEYS:
        check_resources();bound=next(r for r in freeze['cases'] if r['source']==key);base=OUT/f'run185/source_{key}'
        for name in ('profiles.json','surfaces.json','surface_support.npz'):
            assert digest(base/name)==bound['source_files'][name]
        profiles=read(base/'profiles.json')
        with np.load(OUT/'run285'/bound['geometry_file']) as z:offsets=z['offsets'];spans=z['spans']
        rep,prep,merges=normalize(profiles,offsets,spans)
        np.savez(dest/f'source_{key}_alias_map.npz',fragment_representative=rep,profile_representative=prep,
                 canonical_fragments=np.unique(rep))
        write(dest/f'source_{key}_proven_aliases.json',dict(merges=merges,
              ordered_support_identity_proof='same ordered face keys in the same frozen complete surface_support.npz',
              surface_support_sha256=bound['source_files']['surface_support.npz'],
              generator_code_sha256=digest(ROOT/'weld_pipeline/sjg_finite_envelope_carriers_941.py')))
        rows.append(dict(source=key,original_fragments=len(rep),canonical_fragments=len(np.unique(rep)),
                   merged_profiles=len(merges),map_sha256=digest(dest/f'source_{key}_alias_map.npz'),
                   proof_sha256=digest(dest/f'source_{key}_proven_aliases.json'),labels_read=0))
        print('PROVEN_INPUT_ALIASES_FROZEN',key,len(rep),'->',len(np.unique(rep)),flush=True)
        del profiles,rep,prep,spans;gc.collect()
    write(dest/'all_alias_maps_frozen_before_labels.json',dict(cases=rows,labels_read=0,
          code_sha256=digest(Path(__file__)),different_face_or_side_interpretations_never_merged=True,utc=utc()))
    audits=[]
    for key in KEYS:
        bound=next(r for r in rows if r['source']==key)
        assert digest(dest/f'source_{key}_alias_map.npz')==bound['map_sha256']
        with np.load(dest/f'source_{key}_alias_map.npz') as z:rep=z['fragment_representative'];canonical=z['canonical_fragments']
        witness=read(OUT/f'run285/source_{key}_joint_witnesses.json')['witnesses'];occupied={};collisions=[]
        for parent in witness:
            for child in parent['children']:
                for fragment in child['fragments']:
                    token=(int(rep[fragment]),int(child['side_hypothesis']))
                    if token in occupied:collisions.append(dict(canonical_token=token,first=occupied[token],
                                                               second=[parent['parent'],child['child']]))
                    occupied[token]=[parent['parent'],child['child']]
        row=dict(source=key,parents=len(witness),children=sum(len(p['children']) for p in witness),
                 witness_reindex_collisions=collisions,remaining_feature_supervision_conflicts=None)
        if key in ('21','BM79A_native_local','10'):
            folder=OUT/f'run287/source_{key}';fr=read(folder/'receipt.json')
            assert digest(folder/'object_features.npy')==fr['features_sha256']
            f=np.load(folder/'object_features.npy',mmap_mode='r');n=len(canonical)
            x=np.empty((n*2,128),np.float32);x[0::2]=f[canonical];x[1::2]=side_rows(f,canonical,np.ones(n,np.int8))
            y=np.zeros(n*2,np.int8);inverse_fragment={int(v):i for i,v in enumerate(canonical)}
            for fragment,side in occupied:y[inverse_fragment[fragment]*2+side]=1
            values=x.view(np.dtype((np.void,128*4))).ravel();_,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
            positive=np.bincount(inverse,weights=y,minlength=len(counts)).astype(int);conflict=(positive>0)&(positive<counts)
            examples=[]
            for group in np.flatnonzero(conflict)[:5]:
                members=np.flatnonzero(inverse==group)
                examples.append(dict(objects=[[int(canonical[i//2]),int(i%2),int(y[i])] for i in members],
                                     feature_equal=True,distinct_proven_input_identities=True))
            row.update(remaining_feature_supervision_conflicts=int(conflict.sum()),conflict_examples=examples,
                       positive_responsibilities=int(y.sum()))
            del f,x,y,values,inverse,counts;gc.collect()
        audits.append(row);print('ALIAS_REINDEX_CHECK',key,'COLLISIONS',len(collisions),'FEATURE_CONFLICTS',row['remaining_feature_supervision_conflicts'],flush=True)
    passed=all(not r['witness_reindex_collisions'] and not r['remaining_feature_supervision_conflicts'] for r in audits)
    write(dest/'receipt.json',dict(status='ALIAS_NORMALIZATION_AND_WITNESS_REINDEX_PASSED' if passed else 'SUPERVISION_REPRESENTATION_CONFLICT_STOP',
        cases=audits,parents=sum(r['parents'] for r in audits),children=sum(r['children'] for r in audits),
        scientific_training=False,optimizer_updates=0,extraction_correct_rate=None,extraction_precision_mm=None,
        no_source_GT_or_alias_ID_as_feature=True,new_capacity_solver_runs=0,utc=utc()))


if __name__=='__main__':main()
