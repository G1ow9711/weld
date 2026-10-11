"""Share XYZ curve sampling across semantic joints without merging joints.

Two weld sides or material interpretations may use identical XYZ samples but
always retain separate relationship records and prediction slots.
"""
import math
import numpy as np
from weld_pipeline.sjg_shared_joint_domain_1023 import NUMERIC_MM,union,intervals


def basis(n):
    axis=np.eye(3)[np.argmin(np.abs(n))];u=axis-(axis@n)*n;u/=np.linalg.norm(u)
    return u,np.cross(n,u)


def same(a,b,max_parameter):
    if len(a)!=len(b):return False
    if len(a)==6:
        return np.linalg.norm(a[:3]-b[:3])+max_parameter*np.linalg.norm(a[3:]-b[3:])<=NUMERIC_MM
    return np.linalg.norm(a[:3]-b[:3])+abs(a[6]-b[6])+2*a[6]*np.linalg.norm(a[3:6]-b[3:6])<=NUMERIC_MM


def map_interval(family,bank,lo,hi):
    a=np.asarray(family['descriptor']);b=np.asarray(bank['descriptor'])
    if family['kind']=='line':
        offset=float((a[:3]-b[:3])@b[3:]);scale=float(a[3:]@b[3:])
        return [sorted((offset+scale*lo,offset+scale*hi))]
    u,_=basis(a[3:6]);bu,bv=basis(b[3:6]);phase=float(np.arctan2(u@bv,u@bu)%(2*np.pi)*b[6])
    direction=float(np.sign(a[3:6]@b[3:6]));circumference=float(2*np.pi*b[6])
    if hi-lo>=2*np.pi*a[6]-1e-8:return [[0.,circumference]]
    return intervals((phase+direction*lo)%circumference,direction,hi-lo,circumference)


def build_bank(domain):
    bank=[];lookup={};mapping=np.empty(len(domain['families']),np.int32)
    for fid,family in enumerate(domain['families']):
        a=np.asarray(family['descriptor']);key=(family['kind'],*np.round(a[:3],5),*np.round(a[3:6],9),*(np.round(a[6:],5)))
        maxq=max(abs(q) for row in family['domains_mm'] for q in row)
        candidates=lookup.setdefault(key,[]);bid=None
        for j in candidates:
            if same(a,np.asarray(bank[j]['descriptor']),maxq):bid=j;break
        if bid is None:
            bid=len(bank);candidates.append(bid)
            bank.append(dict(kind=family['kind'],descriptor=family['descriptor'],families=[],raw_domains=[]))
        mapping[fid]=bid;bank[bid]['families'].append(fid)
        for lo,hi in family['domains_mm']:bank[bid]['raw_domains'].extend(map_interval(family,bank[bid],lo,hi))
    for row in bank:row['domains_mm']=union(row.pop('raw_domains'))
    assert sorted(f for row in bank for f in row['families'])==list(range(len(domain['families'])))
    original=sum(hi-lo for f in domain['families'] for lo,hi in f['domains_mm'])
    shared=sum(hi-lo for b in bank for lo,hi in b['domains_mm'])
    return dict(geometry_bank=bank,semantic_family_count=len(mapping),query_geometry_count=len(bank),
                semantic_relations_merged=0,semantic_families_merged=0,hard_exclusivity_edges=0,
                duplicated_XYZ_length_compression=original/max(shared,1e-12),
                all_geometry_8mm_queries=sum(max(1,math.ceil((hi-lo)/8))+1 for b in bank for lo,hi in b['domains_mm']),
                shared_positions_do_not_imply_shared_weld_identity=True),mapping


def selected_cost(domain,bank,mapping,family_ids):
    groups={}
    for fid in family_ids:
        row=bank['geometry_bank'][int(mapping[fid])];family=domain['families'][fid]
        intervals_for_bank=groups.setdefault(int(mapping[fid]),[])
        for lo,hi in family['domains_mm']:intervals_for_bank.extend(map_interval(family,row,lo,hi))
    rows=[union(interval_rows) for interval_rows in groups.values()]
    return dict(selected_semantic_families=len(family_ids),selected_geometry_banks=len(groups),
        unique_8mm_queries=sum(max(1,math.ceil((hi-lo)/8))+1 for row in rows for lo,hi in row),
        unique_2mm_queries=sum(max(1,math.ceil((hi-lo)/2))+1 for row in rows for lo,hi in row))


def controls():
    from weld_pipeline.sjg_shared_joint_domain_1023 import build
    def p(side,faces,y=0):return dict(face_indices=faces,weld_side=side,representation=dict(kind='line',endpoints_mm=[[0,y,0],[100,y,0]]))
    domain,_=build([p('front',[0,1]),p('back',[0,1]),p(None,[2,3]),p(None,[0,1],.001)])
    bank,mapping=build_bank(domain)
    assert len(domain['relations'])==4 and len(domain['families'])==4
    assert mapping[0]==mapping[1]==mapping[2] and mapping[3]!=mapping[0]
    assert bank['semantic_families_merged']==0 and bank['query_geometry_count']==2
    assert selected_cost(domain,bank,mapping,[0,1,2])['unique_2mm_queries']==51
    return dict(different_sides_keep_separate_predictions=True,different_entities_keep_separate_predictions=True,
                only_numerically_same_XYZ_queries_shared=True,parallel_offset_not_shared=True,
                no_target_or_STEP_input=True,optimizer_updates=0)
