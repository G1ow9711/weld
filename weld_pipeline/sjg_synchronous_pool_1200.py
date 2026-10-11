"""Frozen run369 state pools and synchronous gradients, no inference changes."""
import copy
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_prefix_minibatch_1189 import partition,mask_digest,parameter_digest
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
from weld_pipeline.sjg_pairwise_oracle_loss_1181 import pairwise_oracle_loss
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache


class ReadOnlyColumns(RelationColumnCache):
    """Existing hardlinked columns are opened rb and copied to read-only RAM."""
    def column(self,key,index,context):
        slot=(key,int(index));path=self.root/key/(str(index)+'.npy')
        if slot not in self.memory and path.exists():
            receipt=read(path.with_suffix('.json'));assert digest(path)==receipt['sha256']
            with path.open('rb') as stream:value=np.load(stream,allow_pickle=False)
            assert value.shape==(len(context.parents),16) and value.dtype==np.float64 and np.isfinite(value).all()
            value.setflags(write=False);self.stats['disk_hits']+=1
            if value.nbytes<=self.memory_limit:
                while self.memory and self.memory_bytes+value.nbytes>self.memory_limit:
                    _,old=self.memory.popitem(last=False);self.memory_bytes-=old.nbytes
                self.memory[slot]=value;self.memory_bytes+=value.nbytes
            else:self.stats['uncached_columns']+=1
            return value
        value=super().column(key,index,context);value.setflags(write=False);return value


def verify_columns_unchanged(source,keys):
    count=0
    for key in keys:
        for path in (source/key).glob('*.npy'):
            receipt=path.with_suffix('.json')
            if receipt.exists():assert digest(path)==read(receipt)['sha256'];count+=1
    return dict(status='SOURCE_COLUMNS_SHA_UNCHANGED',columns=count,
        read_mode='rb into read-only owned RAM, no live column mmap',linked_targets_overwritten=False)


def freeze_pool(dest,keys,bindings):
    source=OUT/'run369';training=read(source/'selector_training.json')
    assert training['bindings']==bindings and training['new_updates']==112
    assert digest(source/'selector.npz')==training['selector_sha256']
    folder=dest/'trajectories';folder.mkdir();entries=[]
    for e in range(4):
        for key in keys:
            path=source/f'trajectories/pass_{e+1:02}_{key}.json';pack=read(path);sha=digest(path)
            rows=[r for r in training['trace'][-112:] if r['stage_epoch']==e+1 and r['source']==key]
            assert len(rows)==4 and all(r['trajectory_sha256']==sha for r in rows)
            assert pack['candidate_binding']==bindings[key] and pack['source']==key and pack['traversal']==e
            assert [v['alpha'] for v in pack['entries']]==[.25,.25,.125,.125,.125,.125]
            assert [(v['origin'],v['kind']) for v in pack['entries']]==[(o,k) for o in ('fresh','run367','run368') for k in ('clean','recovery')]
            for item in pack['entries']:
                t=item['trajectory'];T=t['prefix_count'];assert T==len(t['choices'])+1 and len(set(t['choices']))==T-1
                assert t['source']==key and t['traversal_zero_based']==e
                batches,counts=partition(T,e);assert batches==t['batches'] and counts.tolist()==t['prefix_occurrences']
                if item['origin']!='fresh':assert digest(item['historical_file'])==item['historical_sha256']
            import shutil
            target=folder/path.name;shutil.copy2(path,target);assert digest(target)==sha
            entries.append(dict(source=key,traversal=e,path=str(target),sha256=sha,original_path=str(path)))
    write(dest/'fixed_pool_manifest.json',dict(status='ALL_28_ORIGINAL_SIX_TRAJECTORY_POOLS_FROZEN',entries=entries,
        new_trajectories_generated=0,source_run='run369',selection_by_metrics=False))
    return entries


def pool_for(dest,key,e):
    row=next(v for v in read(dest/'fixed_pool_manifest.json')['entries'] if v['source']==key and v['traversal']==e)
    assert digest(row['path'])==row['sha256']
    return read(row['path']),row['sha256']


def full_pool(model,u,parents,metadata,compat,key,cache,pack,gradient=False):
    loss=0.;illegal=0;premature=0;states=0;weighted_illegal=0.
    g={k:np.zeros_like(v) for k,v in model.parameters.items()} if gradient else None
    before=parameter_digest(model)
    for item in pack['entries']:
        t=item['trajectory'];choices=t['choices'];weight=item['alpha']/t['prefix_count']
        context=NearestParentRelations(parents,metadata,key,cache)
        oracle=ParentSetOracle(compat['compatible_targets'],compat['original_parents'])
        for prefix in range(len(choices)+1):
            check_resources();available,allowed=oracle.action_masks()
            assert mask_digest(available,allowed)==t['mask_sha256'][prefix]
            scores,forward=model.forward(u,context.features(),context.selected)
            value,delta=pairwise_oracle_loss(scores,available,allowed);loss+=weight*value
            choice=model.choose_next(scores,context.selected);action=len(scores)-1 if choice is None else choice
            bad=not bool(allowed[action]);illegal+=int(bad);weighted_illegal+=weight*bad
            premature+=int(choice is None and not allowed[-1]);states+=1
            if gradient:
                local=model.backward(forward,delta)
                for k in g:g[k]+=weight*local[k]
            if prefix==len(choices):assert allowed[-1];break
            chosen=choices[prefix]
            if t.get('kind')=='recovery':
                assert available[chosen];assert oracle.select(chosen)==t['actions'][prefix]['applied_augments']
            else:assert allowed[chosen] and oracle.select(chosen)
            context.add(chosen)
    assert parameter_digest(model)==before
    return dict(source=key,loss=float(loss),illegal_top1=illegal,weighted_illegal_top1=float(weighted_illegal),
        premature_STOP=premature,prefixes_with_trajectory_multiplicity=states),g


def average_gradients(gradients):
    assert len(gradients)==7
    result={k:np.zeros_like(v,dtype=np.float64) for k,v in gradients[0].items()}
    for gradient in gradients:
        for k in result:result[k]+=gradient[k]
    for k in result:result[k]/=7.
    return result


def isolated_copy(model,optimizer):
    return copy.deepcopy(model),copy.deepcopy(optimizer)
