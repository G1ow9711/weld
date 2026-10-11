"""Offline CAD material teacher -> XYZ-only relation pilot, leave-one-workpiece-out."""
from pathlib import Path
import pickle,time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from weld_pipeline.server_surface_pair_features_v2 import local_geometry,pair_features
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild';OUT=BASE/'surface_material_relation_boundary_v2'
def main():
 OUT.mkdir(exist_ok=False);rng=np.random.default_rng(1031);data={};reports=[]
 write(OUT/'protocol.json',dict(keys=['10','25','36'],seed=1031,neighbors=32,random_neighbors=32,normal_knn=32,trees=64,max_depth=16,min_samples_leaf=2,threshold=.5,labels='Exact unique CAD volume equality; unknown endpoints masked; different volume is NOT no-weld.',features='10 symmetric XYZ-only distance/unsigned PCA normal/projection/spectrum features; no absolute coordinates or IDs',evaluation='Leave one workpiece out, teacher pilot only; three previously used TRAIN families, not independent final acceptance',feature_module_sha256=digest(ROOT/'weld_pipeline/server_surface_pair_features_v2.py')))
 for key in ['10','25','36']:
  teacher=BASE/'exact_step_surface_teacher_v3'/key/'teacher.npz';z=np.load(teacher);points=z['xyz_mm'];volume=z['volume_id'];rawpath=BASE/'inputs_v1'/key/'xyz_mm.npy';raw=np.load(rawpath,mmap_mode='r');normals,spectrum=local_geometry(raw,points);raw._mmap.close();near=cKDTree(points).query(points,k=33)[1][:,1:];random=rng.integers(len(points),size=(len(points),32));candidates=np.column_stack((np.repeat(np.arange(len(points)),64),np.column_stack((near,random)).ravel()));candidates=np.unique(np.sort(candidates,axis=1),axis=0);keep=(candidates[:,0]!=candidates[:,1])&(volume[candidates]>=0).all(1);pairs=candidates[keep];x=pair_features(points,normals,spectrum,pairs);y=(volume[pairs[:,0]]==volume[pairs[:,1]]).astype(np.int64);assert np.allclose(x,pair_features(points,-normals,spectrum,pairs[:,::-1]),atol=1e-6);Q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);assert np.allclose(x,pair_features(points@Q.T+17,normals@Q.T,spectrum,pairs),atol=1e-5)
  np.savez(OUT/(key+'_pairs.npz'),features=x,labels=y,point_pairs=pairs,point_indices=z['point_indices']);data[key]=(x,y);reports.append(dict(key=key,pairs=len(y),same_material=int(y.sum()),different_material=int((y==0).sum()),teacher_sha256=digest(teacher),raw_sha256=digest(rawpath)));print(reports[-1],flush=True)
 scores=[]
 for held in ['10','25','36']:
  keys=[k for k in data if k!=held];x=np.concatenate([data[k][0] for k in keys]);y=np.concatenate([data[k][1] for k in keys]);weights=np.concatenate([1/(2*len(keys)*np.maximum(np.bincount(data[k][1],minlength=2)[data[k][1]],1)) for k in keys]);model=RandomForestClassifier(n_estimators=64,max_depth=16,min_samples_leaf=2,max_features='sqrt',random_state=1031,n_jobs=4);model.fit(x,y,sample_weight=weights);test,truth=data[held];pred=model.predict_proba(test)[:,list(model.classes_).index(1)]>=.5;tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel();row=dict(held=held,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),same_recall=float(tp/max(1,tp+fn)),different_false_merge=float(fp/max(1,fp+tn)));scores.append(row);print(row,flush=True)
  olddata=np.load(BASE/'surface_material_relation_pilot_v1'/(held+'_pairs.npz'));oldtruth=olddata['labels'];oldpred=model.predict_proba(olddata['features'])[:,list(model.classes_).index(1)]>=.5;row['original_fixed_pair_confusion']=confusion_matrix(oldtruth,oldpred,labels=[0,1]).tolist()
  with (BASE/'surface_material_relation_pilot_v1'/('leave_'+held+'.pkl')).open('rb') as f:oldmodel=pickle.load(f)
  old_on_new=oldmodel.predict_proba(test)[:,list(oldmodel.classes_).index(1)]>=.5;row['old_model_on_boundary_pairs']=confusion_matrix(truth,old_on_new,labels=[0,1]).tolist()
  with (OUT/('leave_'+held+'.pkl')).open('wb') as f:pickle.dump(model,f)
 write(OUT/'receipt.json',dict(status='EXPERIMENTAL_MATERIAL_RELATION_NOT_DEPLOYED',data=reports,scores=scores,fit_calls=3,GT_weld_reads=0,DEV_reads=0,invariance_controls=True,source_sha256=digest(__file__),scope='Geometric auxiliary relation only, not weld classification/assembly or full extraction. Sparse registered teacher samples and previous TRAIN families limit conclusions.'))
if __name__=='__main__':main()
