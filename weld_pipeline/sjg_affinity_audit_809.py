"""TRAIN-only audit of frozen PTv3 local links, including crop seams."""
import argparse
import gc
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import OUT,TRAIN
from weld_pipeline.sjg_train_776 import check_resources,utc


def summarize(values):
    return dict(count=len(values),quantiles=np.quantile(values,[.01,.1,.5,.9,.95,.99]).tolist()) if len(values) else dict(count=0,quantiles=None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(stage='TRAIN_ONLY_PTV3_LINK_AUDIT',sources=TRAIN,
        sampled_anchors_per_source=8192,neighbors=16,source_sha256=digest(__file__),
        valid_or_arc_labels_read=False,optimizer_updates=0,contract_pass=False,start_utc=utc()))
    frozen=[]
    for si in TRAIN:
        check_resources();xyz=np.load(OUT/'run22'/f'source_{si:02d}_xyz_mm.npy')
        anchors=np.random.default_rng(809+si).choice(len(xyz),min(8192,len(xyz)),replace=False).astype(np.int64)
        neighbors=cKDTree(xyz).query(xyz[anchors],k=17,workers=2)[1][:,1:].astype(np.int64)
        a=np.repeat(anchors,16);b=neighbors.ravel();delta=xyz[b]-xyz[a]
        with np.load(OUT/'run27'/f'source_{si:02d}_features.npz') as pred:
            ed=np.linalg.norm(pred['embedding'][a]-pred['embedding'][b],axis=1)
            cosine=np.abs((pred['normal'][a]*pred['normal'][b]).sum(1))
            edge=np.maximum(pred['boundary'][a],pred['boundary'][b])
        core=np.full(len(xyz),-1,np.int32)
        with np.load(OUT/'run26'/f'blocks_{si:02d}/input_blocks.npz') as blocks:
            for key in blocks.files:
                if key.startswith('ids_'):
                    index=int(key[4:]);core[blocks[key][:int(blocks[f'core_count_{index}'])]]=index
        assert (core>=0).all()
        file=dest/f'source_{si:02d}_pairs.npz'
        np.savez_compressed(file,a=a,b=b,delta=delta,embedding_distance=ed,normal_cosine=cosine,
            boundary_max=edge,cross_core=core[a]!=core[b])
        frozen.append(dict(source=si,file=file.name,sha256=digest(file)))
        print('INPUT_LINKS_FROZEN',si,len(a),flush=True);del xyz,core;gc.collect()
    write(dest/'all_train_links_frozen.json',dict(sources=frozen,teacher_reads_before_freeze=0))
    rows=[]
    for record in frozen:
        si=record['source']
        with np.load(dest/record['file']) as pair,np.load(OUT/'run22'/f'source_{si:02d}_teacher.npz') as teacher:
            a,b=pair['a'],pair['b'];ids=teacher['instance'];known=(ids[a]>=0)&(ids[b]>=0)
            same=known&(ids[a]==ids[b]);different=known&(ids[a]!=ids[b])
            aligned=np.abs((teacher['normal'][a]*teacher['normal'][b]).sum(1))>.999
            coplanar=aligned&(np.abs((pair['delta']*teacher['normal'][a]).sum(1))<.2)
            groups={'same_face_same_core':same&~pair['cross_core'],'same_face_cross_core':same&pair['cross_core'],
                'different_noncoplanar':different&~coplanar,'different_coplanar':different&coplanar}
            row=dict(source=si,unknown_pairs=int((~known).sum()),groups={key:summarize(pair['embedding_distance'][mask]) for key,mask in groups.items()},thresholds=[])
            for threshold in (.2,.4,.6,.8,1.,1.5):
                accepted=(pair['embedding_distance']<=threshold)&(pair['normal_cosine']>=np.cos(np.deg2rad(20)))
                row['thresholds'].append(dict(embedding_threshold=threshold,**{
                    key:dict(accepted=int((accepted&mask).sum()),total=int(mask.sum())) for key,mask in groups.items()}))
        rows.append(row);print('TRAIN_AFFINITY_AUDIT',si,row['groups'],flush=True)
    write(dest/'receipt.json',dict(status='TRAIN_ONLY_LINK_AUDIT_COMPLETE',rows=rows,
        end_utc=utc(),optimizer_updates=0,contract_pass=False,sampled_links_not_exhaustive=True))


if __name__=='__main__':main()
