"""Pure-input ordered finite-contact cells. No learned recursive stopping gate."""
import numpy as np
from weld_pipeline.sjg_shared_joint_domain_1023 import intervals,union
from weld_pipeline.sjg_hierarchical_features_1029 import positions

def family_cells(family,profiles,transforms,extra_anchors=()):
    anchors=list(extra_anchors);observed=[];gaps=[];circ=family['circumference_mm']
    for p in family['profiles']:
        start,sign,length=transforms[p]
        for lo,hi in intervals(float(start),float(sign),float(length),circ):anchors.extend((lo,hi))
        for key,target in [('support_intervals_mm',observed),('unobserved_gaps_mm',gaps)]:
            for a,b in profiles[p].get(key,[]):
                for lo,hi in intervals(float(start+sign*a),float(sign),float(b-a),circ):
                    anchors.extend((lo,hi));target.append((lo,hi))
    observed=union(observed);gaps=union(gaps);cells=[];chains=[]
    for low,high in family['domains_mm']:
        points=np.unique([float(low),float(high)]+[float(q) for q in anchors if low<q<high])
        begin=len(cells)
        for left,right in zip(points[:-1],points[1:]):
            assert right>left
            mid=(left+right)/2
            supported=any(a<=mid<=b for a,b in observed)
            gap_possible=any(a<mid<b for a,b in gaps)
            cells.append((left,right,float(supported),float(gap_possible)))
        chains.append((begin,len(cells)))
    cyclic=bool(circ and len(family['domains_mm'])==1 and abs(family['domains_mm'][0][0])<1e-8 and abs(family['domains_mm'][0][1]-circ)<1e-8)
    return np.asarray(cells,np.float64).reshape(-1,4),chains,cyclic

def build(domain,profiles,transforms,selected_families,boundary_provider=None):
    family_ids=[];cell_rows=[];sequences=[];query_ids=[];query_xyz=[];query_lookup={}
    for fid in selected_families:
        fid=int(fid);family=domain['families'][fid]
        extra=boundary_provider.family(family,domain['relations'][family['relation']]['ordered_face_indices']) if boundary_provider else ()
        cells,chains,cyclic=family_cells(family,profiles,transforms,extra)
        offset=len(cell_rows)
        for first,last in chains:sequences.append(dict(family=fid,start=offset+first,end=offset+last,cyclic=cyclic))
        for left,right,support,gap in cells:
            q=np.asarray([left,(left+right)/2,right]);xyz=positions(family,q);xyz[xyz==0]=0
            row=[]
            for value in xyz:
                key=tuple(float(v) for v in value)
                if key not in query_lookup:query_lookup[key]=len(query_xyz);query_xyz.append(value.tolist())
                row.append(query_lookup[key])
            query_ids.append(row);family_ids.append(fid);cell_rows.append((left,right,support,gap))
    arrays=dict(family=np.asarray(family_ids,np.int64),cells=np.asarray(cell_rows,np.float64).reshape(-1,4),
        query_ids=np.asarray(query_ids,np.int64).reshape(-1,3),query_xyz=np.asarray(query_xyz,np.float64).reshape(-1,3))
    metadata=dict(sequences=sequences,semantic_tokens=len(cell_rows),unique_XYZ_queries=len(query_xyz),
        boundary_slots_per_cell=2,anchor_sources=['carrier finite endpoints','projected observed-support endpoints','observed-gap endpoints'],
        curvature_change='each line/circle has constant curvature; changes at carrier endpoints',
        projected_PT_boundary_points_not_added_in_this_first_constructor=boundary_provider is None,
        no_uniform_dense_grid=True,no_learned_root_pruning=True,semantic_families_merged=0,
        query_sharing='exact XYZ equality with identical original-point feature-neighbour query configuration',target_reads=0)
    if boundary_provider:
        metadata['anchor_sources'].append('projected frozen PTv3 boundary-evidence run endpoints')
        metadata['boundary_anchor_evidence']=boundary_provider.metadata()
    return metadata,arrays

def controls():
    f=dict(kind='line',descriptor=[0,0,0,1,0,0],profiles=[0],domains_mm=[[0,10000]],circumference_mm=0)
    p=[dict(support_intervals_mm=[[0,4000],[6000,10000]],unobserved_gaps_mm=[[4000,6000]])]
    t=np.asarray([[0.,1.,10000.]])
    c,_,_=family_cells(f,p,t);assert c[:,0].tolist()==[0,4000,6000] and c[:,1].tolist()==[4000,6000,10000]
    assert c[1,2]==0 and c[1,3]==1
    f2=dict(f);f2['profiles']=[1]
    meta,a=build(dict(families=[f,f2]),p+p,np.r_[t,t],[0,1])
    assert meta['semantic_tokens']==6 and meta['unique_XYZ_queries']==7 and np.array_equal(a['query_ids'][:3],a['query_ids'][3:])
    return dict(long_structureless_cells_not_densified=True,positive_observed_gap_cell_preserved=True,
        different_semantic_families_keep_tokens=True,exact_XYZ_queries_shared=True,optimizer_updates=0)
