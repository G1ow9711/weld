"""Free native-input line/arc event prediction; no weld target reads."""
import argparse
import gc
import time
import traceback
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_local_training_data_918 import load_case
from weld_pipeline.sjg_local_graph_data_916 import queries,CanonicalQueryTree
from weld_pipeline.sjg_local_curve_model_917 import LocalCurveEventModel
from weld_pipeline.sjg_local_event_train_930 import as_tensors
from weld_pipeline.sjg_local_curve_supervision_915 import decode_interval
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.contract_curve_geometry import CircularArc


def reverse(rep):
    c=CurveCarrier(rep)
    if c.kind=='line':return dict(kind='line',endpoints_mm=c.ends[::-1].tolist())
    a=c.arc
    b=CircularArc(a.center_mm,a.radius_mm,a.plane_normal,(c.at(c.length)-a.center_mm)/a.radius_mm,-a.sweep_rad,a.is_closed)
    return dict(kind='arc',arc_parameters=b.to_parameters())


def decode(rep,q,p,offset):
    curve=CurveCarrier(rep);closed=bool(curve.kind=='arc' and curve.arc.is_closed);result=[]
    if closed and float(p[:,3].mean())>=.5 and float(p[:,0].mean())>=.5:
        return [dict(representation=rep if p[:,4].mean()>=.5 else reverse(rep),confidence=float(p[:,0].mean()),whole=True)]
    if closed:q=q[:-1];p=p[:-1];offset=offset[:-1]
    def peaks(head):
        if closed:
            return np.flatnonzero((p[:,head]>=.5)&(p[:,head]>=np.roll(p[:,head],1))&(p[:,head]>np.roll(p[:,head],-1)))
        return find_peaks(np.r_[-1.,p[:,head],-1.],height=.5)[0]-1
    starts=peaks(1);ends=peaks(2);used=set();previous_end=-np.inf
    start_positions=[]
    for s in starts:
        lo=float(q[s]+np.clip(10*offset[s,0],-8,8));lo=lo%curve.length if closed else float(np.clip(lo,0,curve.length))
        start_positions.append((lo,int(s)))
    for lo,s in sorted(start_positions):
        if lo<previous_end-1e-6:continue
        options=[]
        for e in ends:
            e=int(e)
            if e in used:continue
            hi=float(q[e]+np.clip(10*offset[e,1],-8,8));hi=hi%curve.length if closed else float(np.clip(hi,0,curve.length))
            if closed and hi<=lo:hi+=curve.length
            if hi>lo+1e-6 and (not closed or hi-lo<curve.length):options.append((hi,e))
        for hi,e in sorted(options):
            selected=(q>=lo)&(q<=hi) if hi<=curve.length else (q>=lo)|(q<=hi-curve.length)
            if not selected.any():continue
            fg=float(p[selected,0].mean())
            if fg<.5:continue
            segment=decode_interval(rep,lo,hi if hi<=curve.length else hi-curve.length,wrap=hi>curve.length)
            forward=bool(p[selected,4].mean()>=.5)
            if not forward:segment=reverse(segment)
            result.append(dict(representation=segment,lo=lo,hi=hi,wrap=hi>curve.length,direction_forward=forward,
                               confidence=float((fg*p[s,1]*p[e,2])**(1/3))))
            used.add(e);previous_end=hi;break
    return result


def geometry_nms(predictions,tolerance=1.):
    if not predictions:return [],[]
    curves=[CurveCarrier(p['representation']) for p in predictions];ends=[]
    for c in curves:
        e=c.at(np.array([0.,c.length]));order=sorted(range(2),key=lambda i:tuple(e[i]));ends.append(e[order].reshape(6))
    tree=cKDTree(ends);suppressed=set();kept=[];details=[]
    for i in sorted(range(len(predictions)),key=lambda k:(-predictions[k]['confidence'],k)):
        if i in suppressed:continue
        kept.append(i);a=curves[i]
        for j in tree.query_ball_point(ends[i],np.sqrt(2)*tolerance+1e-8):
            if j==i or j in suppressed or j in kept or curves[j].kind!=a.kind:continue
            b=curves[j]
            if a.project(b.at(b.grid(1.)))[1].max()>tolerance:continue
            if b.project(a.at(a.grid(1.)))[1].max()>tolerance:continue
            suppressed.add(j);details.append(dict(kept=i,suppressed=j))
    return [dict(predictions[i],before_nms_index=i) for i in kept],details


def grid_count(representation,step_mm=2.):
    curve=CurveCarrier(representation)
    return max(1,int(np.ceil(curve.length/step_mm)))+1


def controls():
    near_integer=dict(kind='line',endpoints_mm=[[0,0,0],[2.+1e-12,0,0]])
    assert grid_count(near_integer)==len(CurveCarrier(near_integer).grid(2.))==3
    rep=dict(kind='line',endpoints_mm=[[0,0,0],[100,0,0]]);q=np.arange(101,dtype=float)
    p=np.zeros((101,5));p[20:81,0]=1;p[[20,50],1]=1;p[[50,80],2]=1;p[:,4]=1
    out=decode(rep,q,p,np.zeros((101,2)))
    assert len(out)==2 and np.array_equal(out[0]['representation']['endpoints_mm'],[[20,0,0],[50,0,0]])
    assert np.array_equal(out[1]['representation']['endpoints_mm'],[[50,0,0],[80,0,0]])
    assert len(geometry_nms(out+out)[0])==2
    a=CircularArc([0,0,0],100,[0,0,1],[1,0,0],2*np.pi,True);rep=dict(kind='arc',arc_parameters=a.to_parameters())
    c=CurveCarrier(rep);q=c.grid(2);p=np.zeros((len(q),5));p[:,4]=1
    s=len(q)-20;e=20;p[s:,0]=1;p[:e+1,0]=1;p[s,1]=1;p[e,2]=1
    out=decode(rep,q,p,np.zeros((len(q),2)));assert len(out)==1 and out[0]['wrap']
    return dict(adjacent_children_keep_internal_endpoint=True,geometric_duplicates_suppressed=True,closed_carrier_wrapped_open_arc=True,
        counts_derived_from_actual_curve_not_cached_length=True)


def main(default_data_run='run116',default_training_run='run117',default_proposal_run='run115',default_feature_run='run106'):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--data-run',default=default_data_run);ap.add_argument('--training-run',default=default_training_run);ap.add_argument('--proposal-run',default=default_proposal_run)
    ap.add_argument('--feature-run',default=default_feature_run)
    ap.add_argument('--joint-selection',action='store_true',help='Input-geometry exclusivity within the same predicted joint.')
    ap.add_argument('--sources',nargs='+',help='Explicit diagnostic subset; never emits a complete-cohort receipt.')
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    data=OUT/args.data_run;training=OUT/args.training_run;proposals=OUT/args.proposal_run;features=OUT/args.feature_run;rows=[];torch.set_num_threads(2)
    protocol=dict(stage='LOCAL_TYPED_EVENT_FREE_PREDICTION',source_sha256=digest(__file__),training_run=args.training_run,
        checkpoint_sha256=digest(training/'final.pt'),data_run=args.data_run,proposal_run=args.proposal_run,feature_run=args.feature_run,grid_step_mm=2.,threshold=.5,
        offset_scale_mm=10.,offset_clip_mm=8.,geometric_nms_mm=1.,no_topK=True,original_eight_G=496,
        target_reads=0,STEP_reads=0,all_outputs_frozen_before_scoring=True,optimizer_updates=0,
        query_neighbors='1mm_global_representatives_mapped_to_original_feature_indices',
        query_source_sha256=digest(ROOT/'weld_pipeline/sjg_local_graph_data_916.py'),
        joint_selection=args.joint_selection,parent_grouping_pending=True,E16_pending=True,full_contract_pass=False,start_utc=utc())
    if args.joint_selection:protocol['joint_selection_source_sha256']=digest(ROOT/'weld_pipeline/sjg_joint_interval_selection_1020.py')
    write(dest/'registered_protocol.json',protocol)
    try:
        write(dest/'controls.json',controls());assert read(training/'receipt.json')['status']=='LOCAL_TYPED_EVENT_TRAINING_COMPLETE'
        state=torch.load(training/'final.pt',map_location='cpu')
        assert state['data_manifest_sha256']==digest(data/'training_manifest.json')
        assert read(data/'registered_protocol.json')['features']==args.feature_run
        material_model=state.get('model_family')=='material_curve_event_953'
        anchored_model=state.get('model_family')=='surface_material_anchored_event_1015'
        structure_model=state.get('model_family') in ('surface_material_joint_985','surface_material_anchored_event_1015')
        query_builder=queries
        if anchored_model:
            from weld_pipeline.sjg_boundary_anchored_model_1015 import BoundaryAnchoredModel,queries as anchored_queries
            model=BoundaryAnchoredModel();query_builder=anchored_queries
        elif structure_model:
            from weld_pipeline.sjg_surface_material_joint_model_985 import SurfaceMaterialJointModel
            model=SurfaceMaterialJointModel()
        elif material_model:
            from weld_pipeline.sjg_material_curve_model_953 import MaterialCurveEventModel
            model=MaterialCurveEventModel()
        else:model=LocalCurveEventModel()
        model.load_state_dict(state['state_dict']);del state;model.eval()
        protocol.update(model_family='surface_material_anchored_event_1015' if anchored_model else ('surface_material_joint_985' if structure_model else ('material_curve_event_953' if material_model else 'local_curve_event_917')),
            probability_columns=['weld','start','end','closed','forward']+(['same_material_possible','different_material_possible'] if material_model else []))
        if anchored_model:protocol.update(query_schema='local_10_plus_input_boundary_gaussians_32',query_extension_sha256=digest(ROOT/'weld_pipeline/sjg_boundary_anchored_model_1015.py'))
        write(dest/'registered_protocol.json',protocol)
        graphs={r['source_key']:r for r in read(data/'all_graphs_frozen_before_labels.json')['cases']}
        cases=read(OUT/'run98/local_cohort.json')['cases']
        if args.sources:
            assert len(set(args.sources))==len(args.sources) and set(args.sources)<={c['source_key'] for c in cases}
            cases=[c for c in cases if c['source_key'] in args.sources]
            protocol.update(diagnostic_subset=args.sources,complete_cohort=False)
        else:protocol.update(complete_cohort=True)
        write(dest/'registered_protocol.json',protocol)
        with torch.inference_mode():
            for case in cases:
                check_resources();key=case['source_key'];folder=dest/('source_'+key);folder.mkdir();start=time.perf_counter()
                xyz,point,faces,profiles,_=load_case(case,proposals,features);tree=CanonicalQueryTree(xyz)
                binding=graphs[key];assert digest(data/binding['file'])==binding['sha256'];graph=as_tensors(data/binding['file'])
                if structure_model:face_h,_=model.encode_structure(graph)
                else:face_h=model.encode_faces(graph['face_x'],graph['neighbors'],graph['relation'])
                latent=[]
                for begin in range(0,len(profiles),4096):latent.append(model.encode_curves(face_h,graph['pairs'][begin:begin+4096],graph['geometry'][begin:begin+4096]))
                joint=torch.cat(latent);del latent,face_h,graph
                # Preserve the actual material-conditioned curve context for
                # native parent/attribute heads; never substitute old features.
                np.save(folder/'profile_latent.npy',joint.numpy())
                counts=np.asarray([grid_count(p['representation']) for p in profiles],np.int64)
                ptr=np.r_[0,np.cumsum(counts)];total=int(ptr[-1]);np.save(folder/'profile_offsets.npy',ptr)
                q_map=np.lib.format.open_memmap(folder/'q_mm.npy',mode='w+',dtype=np.float64,shape=(total,))
                p_map=np.lib.format.open_memmap(folder/'probability.npy',mode='w+',dtype=np.float32,shape=(total,7 if material_model else 5))
                o_map=np.lib.format.open_memmap(folder/'offset.npy',mode='w+',dtype=np.float32,shape=(total,2))
                pending_nodes=[];pending_q=[];pending_count=0;written=0;predictions=[]
                def flush():
                    nonlocal pending_nodes,pending_q,pending_count,written
                    if not pending_count:return
                    nodes=np.concatenate(pending_nodes);q=np.concatenate(pending_q)
                    patch,query=query_builder(profiles,nodes,q,xyz,point,tree)
                    outputs=model.query_events(joint[torch.from_numpy(nodes)],torch.from_numpy(patch),torch.from_numpy(query));logits,offset=outputs[:2]
                    if not torch.isfinite(logits).all() or not torch.isfinite(offset).all():raise RuntimeError('Non-finite free prediction')
                    end=written+len(q);q_map[written:end]=q;p_map[written:end,:5]=logits.sigmoid().numpy()
                    if material_model:
                        if not torch.isfinite(outputs[2]).all():raise RuntimeError('Non-finite material relation prediction')
                        p_map[written:end,5:7]=outputs[2].sigmoid().numpy()
                    o_map[written:end]=offset.numpy();written=end
                    pending_nodes=[];pending_q=[];pending_count=0
                    if written%131072<len(q):print('FREE_QUERY',key,written,'/',total,flush=True);check_resources()
                for i,profile in enumerate(profiles):
                    q=CurveCarrier(profile['representation']).grid(2.)
                    assert len(q)==counts[i],('Inconsistent curve sampling count',i,len(q),counts[i])
                    for begin in range(0,len(q),4096):
                        chunk=q[begin:begin+4096];pending_nodes.append(np.full(len(chunk),i,np.int64));pending_q.append(chunk);pending_count+=len(chunk)
                        if pending_count>=4096:flush()
                flush();assert written==total;q_map.flush();p_map.flush();o_map.flush()
                for i,profile in enumerate(profiles):
                    lo,hi=int(ptr[i]),int(ptr[i+1])
                    for prediction in decode(profile['representation'],q_map[lo:hi],p_map[lo:hi],o_map[lo:hi]):
                        predictions.append(dict(prediction,profile=i,material_pair=None,weld_type=None))
                kept,suppressed=geometry_nms(predictions)
                before_selection=len(kept);extra_files=[]
                if args.joint_selection:
                    from weld_pipeline.sjg_joint_interval_selection_1020 import select
                    write(folder/'predictions_before_joint_selection.json',kept)
                    kept,selection=select(kept,profiles)
                    write(folder/'joint_selection.json',selection)
                    extra_files=['predictions_before_joint_selection.json','joint_selection.json']
                write(folder/'predictions_before_nms.json',predictions);write(folder/'predictions.json',kept);write(folder/'nms.json',suppressed)
                del q_map,p_map,o_map
                row=dict(source_key=key,original_G=case['original_G'],profiles=len(profiles),queries=total,
                    predictions=len(kept),before_nms=len(predictions),before_joint_selection=before_selection,elapsed_seconds=time.perf_counter()-start,
                    input_sha256=case['input_sha256'],files={f:digest(folder/f) for f in ('predictions.json','predictions_before_nms.json',
                    'profile_offsets.npy','q_mm.npy','probability.npy','offset.npy','profile_latent.npy')+tuple(extra_files)})
                write(folder/'input_output_freeze.json',row);rows.append(row)
                print('FREE_OUTPUT_FROZEN',key,'PREDICTIONS',len(kept),'QUERIES',total,flush=True)
                del xyz,point,faces,profiles,tree,joint,predictions,kept;gc.collect()
        write(dest/'all_predictions_frozen_before_labels.json',dict(sources=rows,target_reads=0,STEP_reads=0))
        write(dest/'receipt.json',dict(status='LOCAL_TYPED_DIAGNOSTIC_PREDICTIONS_FROZEN' if args.sources else 'LOCAL_TYPED_FREE_PREDICTIONS_FROZEN',sources=rows,
            complete_cohort=not bool(args.sources),full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=rows,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
