"""Finite observed-face boundary events, not physical contact or weld decisions."""
import numpy as np
from .server_finite_plane_endpoint_events import FaceEvents
from .sjg_finite_face_domains_1235 import domains
class FaceBoundaryEvents(FaceEvents):
 def events(self,curve):
  q,source=super().events(curve);events=q.tolist();sources=source.tolist()
  for face,(A,b) in self.domains.items():
   n=self.normals[face];offset=self.offsets[face]
   if curve.kind=='line':
    if abs(float(n@curve.direction))>1e-8:continue
    separation=abs(float(n@curve.ends[0])-offset)
   else:
    if 1-abs(float(n@curve.arc.plane_normal))>1e-8:continue
    separation=abs(float(n@curve.arc.center_mm)-offset)
   # 3mm only bounds a candidate search near observed parallel planes.
   # It is neither a contact test nor a weldability/process allowance.
   if separation>3.:continue
   for lo,hi in domains(curve,A,b):
    events.extend((float(lo),float(hi)));sources.extend((face,face))
  return np.asarray(events),np.asarray(sources)
