"""Check source categories on finite outer/hole boundaries and plane crossing."""
from pathlib import Path
import numpy as np
from tools.cache_boundary_event_provenance import describe,COLUMNS
from weld_pipeline.server_sampled_surface_boundary_events import SampledSurfaceBoundaryEvents
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import write,digest
u,v=np.meshgrid(np.arange(-15.,16),np.arange(-15.,16));points=np.column_stack((u.ravel(),v.ravel(),np.zeros(u.size)));points=points[~((abs(points[:,0])<5)&(abs(points[:,1])<5))]
faces=[dict(normal=[0,0,1],offset=0,point_ids_key='face0')];support={'face0':np.arange(len(points))};curve=CurveCarrier(dict(kind='line',endpoints_mm=[[-20,0,0],[20,0,0]]));g=SampledSurfaceBoundaryEvents(points,faces,support);q,x=describe(g,curve,np.array([len(points)]))
for value in (5,15,25,35):assert x[np.isclose(q,value,atol=1e-8,rtol=0),3].max()==1
for value in (15,25):assert not x[np.isclose(q,value,atol=1e-8,rtol=0),:3].any()
for value in (5,35):assert x[np.isclose(q,value,atol=1e-8,rtol=0),2].max()==1
for value in (0,40):assert x[q==value,0].max()==1
# A transverse finite face contributes the crossing category, not sampled-parallel boundary.
y,z=np.meshgrid(np.arange(-3.,4),np.arange(-3.,4));wall=np.column_stack((np.zeros(y.size),y.ravel(),z.ravel()));points2=np.concatenate((points,wall));faces2=faces+[dict(normal=[1,0,0],offset=0,point_ids_key='face1')];support2=dict(support,face1=np.arange(len(points),len(points2)));q2,x2=describe(SampledSurfaceBoundaryEvents(points2,faces2,support2),curve,np.array([len(points),len(wall)]));hit=x2[np.isclose(q2,20,atol=1e-8,rtol=0)];assert hit[:,1].max()==1 and hit[:,6].max()==1
out=Path('server_rebuild/expansion_v1/boundary_event_provenance_v1/controls.json');write(out,dict(status='PASS',hole_boundaries_distinguished_from_convex_boundary=True,finite_transverse_plane_crossing=True,carrier_ends_marked=True,columns=COLUMNS,source_sha256=digest(__file__),scope='Synthetic source provenance, not physical weldability, rigid-transform invariance or extraction accuracy'))
print('PASS',len(q),len(q2))
