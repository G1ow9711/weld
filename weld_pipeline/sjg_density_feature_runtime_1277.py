"""Exact alternate TRAIN feature views; inference must use original features."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest

SEQUENCE_COLUMNS=np.arange(28,43)
ENDPOINT_COLUMNS=np.concatenate([SEQUENCE_COLUMNS+56*i for i in range(3)])


def interpolate_endpoint_view(grid,alternate,q,length,closed):
    grid=np.asarray(grid);q=np.asarray(q)
    assert alternate.shape==(len(grid),15) and np.all(np.diff(grid)>0)
    assert length>0 and np.all((q>=0)&(q<=length))
    sides=[]
    for inward in (1,-1):
        queries=[q,q+inward*50.,q-inward*50.]
        queries=[np.mod(v,length) if closed else np.clip(v,0,length) for v in queries]
        sides.append(np.column_stack([np.interp(v,grid,alternate[:,j]) for v in queries for j in range(15)]))
    return np.concatenate(sides).astype(np.float32)


def overlay_endpoint(raw,alternate):
    assert raw.ndim==2 and raw.shape[1]==180 and alternate.shape==(len(raw),45)
    result=np.array(raw,copy=True);result[:,ENDPOINT_COLUMNS]=alternate
    assert np.isfinite(result).all()
    return result


class DensityView:
    def __init__(self,key,run='run436'):
        manifest=read(OUT/run/'receipt.json')
        assert manifest['status']=='FIVE_TRAIN_DENSITY_FEATURE_VIEWS_FROZEN_NO_FIT'
        if key not in manifest['TRAIN']:raise ValueError('Augmentation is confined to registered TRAIN sources')
        folder=OUT/run/f'source_{key}';r=read(folder/'receipt.json')
        assert r==next(c for c in manifest['cases'] if c['source']==key)
        assert digest(folder/'dimensional_knn_half.npy')==r['alternate_sha256']
        assert digest(folder/'sequence_map.npz')==r['map_sha256']
        with np.load(folder/'sequence_map.npz') as z:self.mapping={k:z[k] for k in z.files}
        self.values=np.load(folder/'dimensional_knn_half.npy')
        assert list(self.values.shape)==r['shape'] and np.isfinite(self.values).all()
        self.lookup={int(pid):i for i,pid in enumerate(self.mapping['profile_ids'])}
        self.receipt=r

    def sequence(self,raw,a,b):
        assert raw.shape==(b-a,56)
        result=np.array(raw,copy=True);result[:,SEQUENCE_COLUMNS]=self.values[a:b]
        return result

    def endpoint(self,row,q,length,closed):
        j=self.lookup[int(row)//2];a,b=map(int,self.mapping['offsets'][j:j+2])
        return interpolate_endpoint_view(self.mapping['q_mm'][a:b],self.values[a:b],q,length,closed)


def controls():
    grid=np.array([0.,50.,100.]);alternate=np.repeat(grid[:,None],15,axis=1)
    q=np.array([0.,25.,100.]);view=interpolate_endpoint_view(grid,alternate,q,100.,False)
    expected=np.array([[0,50,0],[25,75,0],[100,100,50],[0,0,50],[25,0,75],[100,50,100]])
    assert np.array_equal(view.reshape(6,3,15),np.repeat(expected[:,:,None],15,axis=2))
    raw=np.arange(1080,dtype=np.float32).reshape(6,180);saved=raw.copy();changed=overlay_endpoint(raw,view)
    retained=np.setdiff1d(np.arange(180),ENDPOINT_COLUMNS)
    assert len(retained)==135 and np.array_equal(raw,saved)
    assert np.array_equal(changed[:,retained],raw[:,retained])
    wrapped=interpolate_endpoint_view(grid,alternate,np.array([0.,100.]),100.,True)
    assert np.array_equal(wrapped.reshape(4,3,15)[:,:,0],np.array([[0,50,50]]*4))
    return dict(status='PASS',analytic_open_and_closed_queries=True,original_arrays_not_mutated=True,
        unchanged_endpoint_columns=135,changed_endpoint_columns=45,changed_sequence_columns=15)


if __name__=='__main__':print(controls())
