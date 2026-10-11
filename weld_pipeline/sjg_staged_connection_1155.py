"""Unchanged full instance connection construction split into its own process stage."""
import argparse,gc,pickle,time,traceback
from contextlib import ExitStack
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS,feature_path
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case,predict_instances,relative_link_blocks
from weld_pipeline.sjg_interval_forest_train_1130 import instance_edge_truth
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256


def generate(dest,models,phase):
    bindings=[]
    # Freeze every new predicted graph before reading edge association targets.
    for key in KEYS:
        phase('FREEZE_PREDICTED_INSTANCE_CONNECTIONS',source=key)
        case=load_case(key);instances=[v for v in read(OUT/f'run318/source_{key}_interval_instances.json')['instances'] if v['count_is_known']]
        screen=predict_instances(case,models,[(v['row'],v['k']) for v in instances]);nodes=screen['nodes']
        folder=dest/f'links_{key}';folder.mkdir();write(folder/'predicted_nodes.json',screen);m=0
        with (folder/'features.f32').open('xb') as fx,(folder/'links.i32').open('xb') as fl,(folder/'legal.u8').open('xb') as fm:
            for links,features,legal in relative_link_blocks(nodes,case):
                fx.write(features.astype('<f4',copy=False).tobytes());fl.write(links.astype('<i4').tobytes());fm.write(legal.astype('u1').tobytes());m+=len(links)
                if m%100000<len(links):check_resources()
        bindings.append(dict(source=key,nodes=len(nodes),edges=m,invalid_instances=len(screen['invalid']),admitted_instances=screen['admitted_instances'],
            features_sha256=digest(folder/'features.f32'),nodes_sha256=digest(folder/'predicted_nodes.json'),links_sha256=digest(folder/'links.i32'),legal_sha256=digest(folder/'legal.u8')))
        case['x']._mmap.close();del case,nodes,screen,instances;gc.collect();check_resources()
    write(dest/'all_predicted_geometry_frozen_before_edge_labels.json',dict(cases=bindings,GT_endpoint_overrides=0))
    total=0
    with (dest/'link_train.f32').open('xb') as fx,(dest/'link_labels.i8').open('xb') as fy:
        for bound in bindings:
            key=bound['source'];phase('LABEL_COMPLETE_INSTANCE_CONNECTIONS',source=key)
            folder=dest/f'links_{key}';nodes=read(folder/'predicted_nodes.json')['nodes'];m=bound['edges']
            instances=[v for v in read(OUT/f'run318/source_{key}_interval_instances.json')['instances'] if v['count_is_known']]
            owners,truth,compiled=instance_edge_truth(key,instances)
            with np.load(OUT/f'run318/source_{key}_count_targets.npz') as z:count_labels=z['labels']
            ids=[(v['object_row'],v['k']) for v in nodes];seen=set();supervised=0
            if m:
                links=np.memmap(folder/'links.i32',dtype='<i4',mode='r',shape=(m,2));legal=np.memmap(folder/'legal.u8',dtype='u1',mode='r',shape=(m,3))
                features=np.memmap(folder/'features.f32',dtype='<f4',mode='r',shape=(m,32))
                for first in range(0,m,8192):
                    part=links[first:first+8192];labels=np.zeros(len(part),np.int8);known=np.zeros(len(part),bool)
                    for j,(a,b) in enumerate(part):
                        u,v=ids[int(a)],ids[int(b)];kind=truth.get((u,v))
                        if kind is not None:
                            if legal[first+j,kind-1]:labels[j]=kind;known[j]=True;seen.add((u,v))
                            continue
                        ou,ov=owners.get(u),owners.get(v)
                        if count_labels[u[0]]==0 or count_labels[v[0]]==0:known[j]=True
                        elif ou is not None and ov is not None:
                            if ou[0]!=ov[0]:known[j]=True
                            elif ou[1]!=ov[1]:
                                parent=compiled['parents'][ou[0]]
                                adjacent=ov[1]==ou[1]+1 or (parent['closed'] and ou[1]==parent['child_count']-1 and ov[1]==0)
                                if not adjacent:known[j]=True
                    fx.write(np.asarray(features[first:first+len(part)][known]).tobytes());fy.write(labels[known].tobytes());supervised+=int(known.sum())
                for a in (links,legal,features):a._mmap.close()
            misses=[dict(first=list(u),second=list(v),kind=k) for (u,v),k in truth.items() if (u,v) not in seen]
            write(folder/'predicted_geometry_edge_misses.json',dict(misses=misses,GT_endpoint_overrides=0))
            bound.update(supervised_edges=supervised,undetermined_edges=m-supervised,predicted_geometry_true_edge_misses=len(misses));total+=supervised
            del nodes,instances,count_labels,compiled,owners,truth;gc.collect();check_resources()
    return total,bindings
