"""Locally conditioned analytic residual model; pure-PCD inference only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .world_set_model_285 import WorldSetModel,observed_reference_rows
from .world_set_geometry_285 import parent_samples
from .local_point_observation_295 import local_point_features

VERSION="PURE_RAW_LOCAL_ANALYTIC_UNORDERED_AUX_295_v1"

class LocalWorldModel(WorldSetModel):
    def __init__(self,parent_queries=512,child_capacity=16,hidden=128,point_block_size=16384):
        super().__init__(parent_queries,child_capacity,hidden,point_block_size)
        self.local_refinement=nn.Sequential(nn.Linear(hidden+9*39,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.material_encoder=nn.Sequential(nn.Linear(39,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.material_head=nn.Sequential(nn.Linear(hidden+64,128),nn.SiLU(),nn.Linear(128,8))
        with torch.no_grad():
            self.local_refinement[-1].weight.zero_(); self.local_refinement[-1].bias.zero_()
            self.material_head[-1].bias.copy_(torch.tensor([10.,0.,0.,1.,10.,0.,1.,0.]))

    def _outputs(self,parents,reference,origin,input_receipt):
        children=self.sequence_encoder(parents[:,None]+self.sequence_positions.weight[None])
        circle=self.circle_parameters(parents)
        kinds=self.sequence_kind(children)
        # Present-parent grammar: first item line/arc; STOP may begin at item1.
        first=torch.cat((torch.full_like(kinds[:,:1,:1],-1e9),kinds[:,:1,1:]),dim=-1)
        kinds=torch.cat((first,kinds[:,1:]),dim=1)
        return dict(version=VERSION,origin_mm=origin,input_receipt=input_receipt,
            parent_embeddings=parents,child_embeddings=children,
            presence_logits=self.presence(parents).squeeze(-1),mode_logits=self.parent_mode(parents),
            closed_logits=self.parent_closed(parents),direction_logits=self.parent_direction(parents),turns_logits=self.parent_turns(parents),
            line_endpoints_m=self.line_endpoints(parents).reshape(-1,2,3)+reference[:,None],
            circle_center_m=circle[:,:3]+reference,circle_radius_m=F.softplus(circle[:,3]),circle_quaternion=circle[:,4:8],
            circle_phase_rad=circle[:,8],circle_open_sweep_rad=2*np.pi*torch.sigmoid(circle[:,9]),
            sequence_kind_logits=kinds,weld_type_logits=self.sequence_weld_type(children),gap_logits=self.sequence_gap(children),
            interval_weights=F.softplus(self.sequence_interval(children)),
            mixed_endpoints_m=self.mixed_endpoints(children).reshape(self.parent_queries,self.child_capacity+1,2,3)+reference[:,None,None],
            mixed_arc_parameters=self.mixed_arc(children),mixed_direction_logits=self.mixed_arc_direction(children),
            global_parent_query_capacity=self.parent_queries,global_child_sequence_capacity=self.child_capacity)

    def _self_anchors(self,output):
        """Own endpoints plus seven equal-arclength locations, never GT semantics."""
        anchors=np.full((self.parent_queries,9,3),np.nan,np.float64)
        with torch.no_grad():
            for query in range(self.parent_queries):
                mode=int(output["mode_logits"][query].argmax())
                closed=bool(output["closed_logits"][query].argmax()) if mode!=0 else False
                direction=int(output["direction_logits"][query].argmax())*2-1
                kinds=output["sequence_kind_logits"][query].argmax(-1)
                stops=torch.nonzero(kinds==0).flatten()
                count=min(self.child_capacity,int(stops[0])) if len(stops) else self.child_capacity
                count=max(1,count)
                # Analytic parent branch determines its own observation carrier.
                observed_count=1 if mode in (0,1) else count
                branch_kinds=(torch.ones(1,1,dtype=torch.long)*(1 if mode==0 else 2)) if mode in (0,1) else kinds[:count][None]
                gaps=torch.zeros((1,0),dtype=torch.bool) if observed_count==1 else output["gap_logits"][query,:count-1].argmax(-1).bool()[None]
                try:
                    points,parameters=parent_samples(output,torch.tensor([query]),mode,observed_count,gaps,
                        torch.tensor([direction]),torch.tensor([closed]),branch_kinds,sample_count=7)
                    if mode in (0,1):
                        values=points[0,0].numpy()
                        anchors[query]=np.concatenate((values[:1],values,values[-1:]))+output["origin_mm"].numpy()
                    else:
                        values=points[0].numpy()
                        infos=parameters["mixed_arc_parameters"]
                        lengths=np.asarray([float(info["radius"][0]*info["sweep"][0].abs()) if info is not None else np.linalg.norm(v[-1]-v[0]) for v,info in zip(values,infos)])
                        cumulative=np.concatenate(([0.],np.cumsum(lengths)))
                        if cumulative[-1]<=0: continue
                        selected=[]
                        for fraction in (0.,*np.linspace(0.,1.,7),1.):
                            s=fraction*cumulative[-1]
                            child=min(observed_count-1,int(np.searchsorted(cumulative[1:],s,side="left")))
                            f=(s-cumulative[child])/lengths[child]
                            info=infos[child]
                            if f<=0: value=values[child,0]
                            elif f>=1: value=values[child,-1]
                            elif info is None: value=values[child,0]+f*(values[child,-1]-values[child,0])
                            else:
                                angle=float(info["sweep"][0])*f
                                u=info["u"][0].numpy(); normal=info["normal"][0].numpy()
                                value=info["center"][0].numpy()+float(info["radius"][0])*(np.cos(angle)*u+np.sin(angle)*np.cross(normal,u))
                            selected.append(value)
                        anchors[query]=np.asarray(selected)+output["origin_mm"].numpy()
                except (ValueError,RuntimeError,OverflowError):
                    # Invalid prediction receives empty observations, retains FP.
                    continue
        return anchors

    def forward(self,source):
        tokens,reference,origin,receipt=self.encode_original_source(source)
        positions=(source.mean_xyz_mm[source.counts>0]-source.center_mm)/1000.
        rows=torch.from_numpy(observed_reference_rows(positions,self.parent_queries))
        parents=self.parent_decoder((tokens[rows]+self.parent_queries_embedding.weight+self.query_positions(reference))[None],tokens[None])[0]
        coarse=self._outputs(parents,reference,origin,dict(receipt))
        output=coarse
        valid=[]
        for stage in range(2):
            anchors=self._self_anchors(output)
            observations=local_point_features(source,anchors.reshape(-1,3),self.point_block_size).reshape(self.parent_queries,-1)
            valid.append(int((observations.reshape(-1,3,13)[:,:,0].sum(axis=1)>0).sum()))
            parents=parents+self.local_refinement(torch.cat((parents,torch.from_numpy(observations)),dim=-1))
            output=self._outputs(parents,reference,origin,dict(receipt))
        output["coarse_prediction"]=coarse
        output["input_receipt"].update(local_refinement_stages=2,own_prediction_observation_positions=self.parent_queries*9*2,
            nonempty_local_positions_per_stage=valid,local_radii_mm=[16.,64.,256.],local_GT_or_STEP_reads=0,
            point_index_rows=len(source.xyz_mm),auxiliary_branch_detaches_geometry=True)
        return output

    def predict_materials(self,source,child_embeddings,positions_world_mm):
        positions=np.asarray(positions_world_mm,np.float64)
        if positions.ndim!=3 or positions.shape[-1]!=3: raise ValueError("Child x point x worldXYZ required")
        observed=torch.from_numpy(local_point_features(source,positions.reshape(-1,3),self.point_block_size)).reshape(*positions.shape[:2],39)
        embedding=child_embeddings.detach()[:,None].expand(-1,positions.shape[1],-1)
        raw=self.material_head(torch.cat((embedding,self.material_encoder(observed)),dim=-1)).reshape(*positions.shape[:2],2,4)
        thickness=F.softplus(raw[...,0])
        normals=F.normalize(raw[...,1:],dim=-1,eps=1e-8)
        return dict(thickness_mm=thickness,CAD_outward_unit_normals_world=normals,raw_normal_lengths=torch.linalg.vector_norm(raw[...,1:],dim=-1),
            unordered_material_names=["material_A","material_B"],formal_E_status="UNKNOWN_SIDE_CONVENTION",geometry_detached=True)
