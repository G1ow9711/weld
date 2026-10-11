"""Material relation context rebuilt from XYZ and predicted finite support only."""
import gc
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_geometry_776 import digest
from weld_pipeline.sjg_train_776 import check_resources
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_material_runtime_861 import infer_material_relations
from weld_pipeline.sjg_ptv3_runtime_807 import core_partitions
from weld_pipeline.sjg_material_ambiguity_874 import MaterialAmbiguityHead
from weld_pipeline.sjg_relation_features_880 import RelationHead,pair_features
from weld_pipeline.sjg_adjacency_eval_842 import ThirdSurfaceContext

CONTEXT_NAMES=['cross_face_pair_fraction','observed_cross_face_pair_fraction','distance_score_mean','distance_score_max',
    'geometry_score_mean','geometry_score_max','material_ambiguity_mean','material_ambiguity_max']
CHECKPOINTS={'point':'run74/dense_control/last.pt','ambiguity':'run79/last.pt',
    'distance':'run84/distance_only/last.pt','geometry':'run84/finite_geometry/last.pt'}


def load_weights(model,key):
    path=OUT/CHECKPOINTS[key]
    with path.open('rb') as h:state=torch.load(h,map_location='cuda')
    model.load_state_dict(state['model']);model.eval();return digest(path)


class FeatureCapture(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.voxel=None;self.original=None
        self.hook=base.backbone.register_forward_hook(self.capture)

    def capture(self,module,args,output):self.voxel=output.feat

    def forward(self,block):
        result=self.base(block);self.original=self.voxel[torch.from_numpy(block['inverse']).cuda()];return result


def input_support(xyz,faces):
    archives={};rr=[];cc=[];bindings={}
    try:
        for index,face in enumerate(faces):
            path=ROOT/face['support_file'];key=str(path)
            if key not in archives:
                assert digest(path)==face['support_sha256'];archives[key]=np.load(path);bindings[key]=face['support_sha256']
            ids=archives[key][face['point_ids_key']].astype(np.int64);rr.append(ids);cc.append(np.full(len(ids),index,np.int64))
    finally:
        for z in archives.values():z.close()
    rows=np.concatenate(rr);cols=np.concatenate(cc);assert np.all((rows>=0)&(rows<len(xyz)))
    support=csr_matrix((np.ones(len(rows),np.int32),(rows,cols)),shape=(len(xyz),len(faces)))
    support.sum_duplicates();support.data[:]=1;return support,bindings


def material_fields(xyz,faces,progress=None):
    """No source ID, reference geometry or ground-truth argument exists."""
    weights={};base=SurfaceMaterialModel().cuda();weights['point']=load_weights(base,'point');capture=FeatureCapture(base)
    parts=list(core_partitions(xyz));features=np.empty((len(xyz),32),np.float32)
    def record(done,total,count,observed,pairs):
        core=parts[done-1];assert len(core)==count;features[core]=capture.original[:count].cpu().numpy()
        if progress and (done==1 or done%20==0 or done==total):progress('POINT_BLOCK',done,total)
    try:relations=infer_material_relations(capture,xyz,progress=record,resource_check=check_resources)
    finally:capture.hook.remove()
    assert np.isfinite(features).all();del capture,base,parts;gc.collect();torch.cuda.empty_cache()
    head=MaterialAmbiguityHead(np.zeros(32),np.ones(32)).cuda();weights['ambiguity']=load_weights(head,'ambiguity')
    ambiguity=np.empty(len(xyz),np.float32)
    with torch.inference_mode():
        for i in range(0,len(xyz),65536):ambiguity[i:i+65536]=head(torch.from_numpy(features[i:i+65536]).cuda()).sigmoid().cpu().numpy()
    del head,features;gc.collect();torch.cuda.empty_cache();support,support_bindings=input_support(xyz,faces)
    if progress:progress('PAIR_FEATURES',len(xyz),len(faces))
    x,flat=pair_features(xyz,relations['neighbors'],relations['material_distance'],relations['surface_distance'],relations['normal'],
        relations['boundary'],ambiguity,support,relations['core_block']);probabilities=[]
    for key,geometry in [('distance',False),('geometry',True)]:
        head=RelationHead(np.zeros(20),np.ones(20),geometry).cuda();weights[key]=load_weights(head,key)
        values=np.full(relations['neighbors'].shape,np.nan,np.float32)
        with torch.inference_mode():
            for i in range(0,len(x),65536):values.ravel()[flat[i:i+65536]]=head(torch.from_numpy(x[i:i+65536]).cuda()).sigmoid().cpu().numpy()
        probabilities.append(values);del head;gc.collect();torch.cuda.empty_cache()
    return dict(neighbors=relations['neighbors'],probabilities=probabilities,ambiguity=ambiguity,support=support,
        weights=weights,support_bindings=support_bindings,every_original_point_once=bool(np.all(relations['core_coverage']==1)))


class MaterialQueryContext:
    def __init__(self,xyz,profiles,fields):
        self.xyz=xyz;self.profiles=profiles;self.fields=fields;self.tree=cKDTree(xyz);self.support=fields['support'].tocsc()

    def __call__(self,profile,q):
        p=self.profiles[profile];left,right=p['face_indices'];n=len(self.xyz)
        a=np.zeros(n,bool);b=np.zeros(n,bool)
        for face,mask in [(left,a),(right,b)]:mask[self.support.indices[self.support.indptr[face]:self.support.indptr[face+1]]]=True
        values=[];fields=self.fields
        for start in range(0,len(q),512):
            qq=q[start:start+512];positions=np.asarray(p['p0'])+qq[:,None]*p['direction']
            _,near=self.tree.query(positions,k=min(16,n),workers=2)
            if near.ndim==1:near=near[:,None]
            neighbor=fields['neighbors'][near];other=neighbor.clip(0)
            cross=(neighbor>=0)&((a[near][...,None]&b[other])|(b[near][...,None]&a[other]))
            observed=cross&np.isfinite(fields['probabilities'][0][near])&np.isfinite(fields['probabilities'][1][near])
            count=observed.sum((1,2));denominator=cross.shape[1]*cross.shape[2]
            columns=[cross.sum((1,2))/denominator,count/denominator]
            for probability in fields['probabilities']:
                score=np.where(observed,probability[near],0.);columns.extend([score.sum((1,2))/np.maximum(1,count),score.max((1,2))])
            amb=fields['ambiguity'][near];columns.extend([amb.mean(1),amb.max(1)])
            values.append(np.stack(columns,1).astype(np.float32))
        result=np.concatenate(values);assert result.shape==(len(q),8) and np.isfinite(result).all()
        assert np.all((result>=0)&(result<=1));return result


class WeldMaterialContext:
    """Free-inference context factory matching the existing sequence interface."""
    def __init__(self,xyz,faces,profiles,selected):
        self.third=ThirdSurfaceContext(xyz,faces,profiles,selected)
        fields=material_fields(xyz,faces,lambda *args:print('WELD_MATERIAL_RUNTIME',*args,flush=True))
        self.material=MaterialQueryContext(xyz,profiles,fields);self.fields=fields

    def __call__(self,profile,q):return np.c_[self.third(profile,q),self.material(profile,q)]

    def describe(self):return dict(third_surface=self.third.describe(),material_features=CONTEXT_NAMES,
        checkpoint_sha256=self.fields['weights'],support_bindings=self.fields['support_bindings'],
        every_original_point_once=self.fields['every_original_point_once'],no_source_ID_lookup=True,
        raw_XYZ_point_fields_recomputed=True,reference_labels_not_input=True,material_identity_still_unknown=True)
