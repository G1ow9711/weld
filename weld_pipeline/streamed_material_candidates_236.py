"""Store every unreviewed material candidate without a resident Python list.

Original 204 face/overlap computations and 204b candidate fields are retained.
The storage revision does not certify material identity or customer normals.
"""
from collections.abc import Mapping
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np

from .finite_plate_binding_v1_204 import FinitePlateCandidates as Original204, CandidateConfig
from .finite_plate_binding_v1_204b import FinitePlateCandidates as Original204b

VERSION = 'COMPLETE_MATERIAL_CANDIDATE_ORDERED_GZIP_RECORD_STORAGE_236_v1'


def file_sha(path):
    value=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


class GzipCandidateStore:
    def __init__(self,path,on_append=None):
        self.path=Path(path);self.temporary=self.path.with_name(self.path.name+'.part')
        if self.path.exists() or self.temporary.exists():
            raise FileExistsError('Candidate storage is immutable and cannot replace existing evidence')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.raw=self.temporary.open('xb')
        self.encoder=gzip.GzipFile(filename='',mode='wb',fileobj=self.raw,mtime=0,compresslevel=6)
        self.count=0;self.descriptor=None;self.complete=False;self.on_append=on_append

    def __len__(self):
        return self.count

    def append(self,record):
        if self.complete or self.encoder.closed:
            raise ValueError('Completed or closed candidate storage cannot be appended')
        if (not isinstance(record,Mapping) or type(record.get('plate_id')) is not int
                or record['plate_id']!=self.count):
            raise ValueError('Every original plate ordinal must be appended exactly once in order')
        text=json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n'
        self.encoder.write(text.encode('utf-8'))
        self.count+=1
        if self.on_append is not None and self.count%256==0:
            self.on_append(self.count)

    def finish(self):
        if self.complete:
            raise ValueError('Candidate storage already committed')
        self.encoder.close();self.raw.flush();os.fsync(self.raw.fileno());self.raw.close()
        attempts=0
        for delay in (0.,.01,.025,.05,.1,.2,.4,.8):
            if delay:
                time.sleep(delay)
            attempts+=1
            try:
                if self.path.exists():
                    raise FileExistsError('Existing material evidence cannot be replaced')
                os.rename(self.temporary,self.path)
                break
            except OSError as exc:
                if getattr(exc,'winerror',None) not in (5,32,33) or attempts==8:
                    raise
        self.complete=True
        self.descriptor=dict(path=str(self.path.resolve()),sha256=file_sha(self.path),
            bytes=self.path.stat().st_size,ordered_complete_candidates=self.count,
            format='GZIP_UTF8_ONE_CANONICAL_JSON_PER_ORIGINAL_PLATE_ORDINAL',
            replace_attempts=attempts,immutable=True,all_candidate_fields_retained=True)
        return self.descriptor

    def __iter__(self):
        if not self.complete:
            raise ValueError('Uncommitted candidate evidence cannot be consumed')
        if file_sha(self.path)!=self.descriptor['sha256']:
            raise ValueError('Candidate stream content binding failed')
        total=0
        with gzip.open(self.path,'rt',encoding='utf-8') as stream:
            for line in stream:
                record=json.loads(line)
                if type(record.get('plate_id')) is not int or record['plate_id']!=total:
                    raise ValueError('Candidate stream lost original ordinal/order')
                if total>=self.count:
                    raise ValueError('Candidate stream has unexpected extra records')
                total+=1
                yield record
        if total!=self.count:
            raise ValueError('Candidate stream is incomplete')

    def close_incomplete(self):
        if not self.encoder.closed:
            self.encoder.close()
        if not self.raw.closed:
            self.raw.flush();self.raw.close()


class StreamedMaterialCandidates(Original204b):
    def __init__(self,xyz_mm,plane_hypotheses,config=CandidateConfig(),*,material_directory,progress=None):
        self.material_directory=Path(material_directory)
        self.material_progress=progress
        self.material_store_bindings={}
        super().__init__(xyz_mm,plane_hypotheses,config)

    def _material_pairs(self):
        notify=(None if self.material_progress is None else
                lambda count:self.material_progress(stage='MATERIAL_PAIR_BASE_ENUMERATION',
                                                    complete_candidates=count))
        base=GzipCandidateStore(self.material_directory/'all_base204_material_pairs.jsonl.gz',notify)
        self.plates=base
        try:
            # Invoke the original complete enumeration/overlap/geometry formula.
            Original204._material_pairs(self)
            self.material_store_bindings['base204']=base.finish()
        except BaseException:
            base.close_incomplete()
            raise
        enriched=GzipCandidateStore(self.material_directory/'all_enriched204b_material_pairs.jsonl.gz')
        try:
            for pair in base:
                left,right=pair['face_ids']
                a,b=self._geometry[left],self._geometry[right]
                signed=self.faces[right]['offset_mm']-a['xyz']@b['normal']
                pair['signed_separation_at_left_support_range_mm']=[float(signed.min()),float(signed.max())]
                pair['thickness_measurement_scope']='CANDIDATE_NORMAL_SEPARATION_NOT_CONTRACT_E2_THICKNESS'
                pair['possible_crossing_within_left_support']=bool(signed.min()<=0.<=signed.max())
                direction=np.sign(signed[0])
                pair['conditional_material_outward_normals']=[
                    (-direction*np.sign(a['normal']@b['normal'])*a['normal']).tolist(),
                    (direction*b['normal']).tolist()]
                pair['signed_customer_normal_is_not_confirmed']=True
                enriched.append(pair)
                if self.material_progress is not None and len(enriched)%256==0:
                    self.material_progress(stage='MATERIAL_PAIR_ENRICHMENT',complete_candidates=len(enriched),
                                           total_candidates=len(base))
            self.material_store_bindings['enriched204b']=enriched.finish()
        except BaseException:
            enriched.close_incomplete()
            raise
        if len(base)!=len(enriched):
            raise ValueError('Enrichment changed the complete material candidate denominator')
        self.plates=enriched

    def to_dict(self):
        result=super().to_dict()
        del result['plate_candidates']
        result.update(version=VERSION,plate_candidates_artifact=self.plates.descriptor,
            material_candidate_count=len(self.plates),material_store_bindings=self.material_store_bindings,
            original204_204b_all_candidate_fields_retained=True,material_identity_verified=False,
            hidden_holes_or_closed_solid_identity_verified=False)
        return result

    @classmethod
    def from_raw_xyz(cls,xyz_mm,config=CandidateConfig(),*,material_directory,progress=None):
        # Same raw generators, seeds, formulas and ordering as the frozen 204b.
        from .robust_surface_planes import surface_candidates
        from .small_finite_surface_events_v3 import small_surface_planes
        xyz=np.asarray(xyz_mm,np.float64)
        if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
            raise ValueError('Complete finite current XYZ required')
        if len(xyz):
            _,geometry=surface_candidates(xyz)
            vertical,_=small_surface_planes(xyz,geometry)
            hypotheses=[dict(normal=[*p['normal_xy'],0.],offset_mm=p['offset_mm']) for p in vertical]
            hypotheses.extend(dict(normal=[0.,0.,1.],offset_mm=z) for z in geometry['horizontal_levels'])
        else:
            hypotheses=[]
        return cls(xyz,hypotheses,config,material_directory=material_directory,progress=progress)
