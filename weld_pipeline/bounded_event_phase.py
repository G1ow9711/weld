"""Generic physical query phases and bounded event-center residuals."""
import numpy as np

from weld_pipeline.near_event_votes import RADIUS_MM
from weld_pipeline.neural_physical_event_votes import point_features

VERSION='bounded_physical_cell_center_features_v1'
CELL_MM=4.*RADIUS_MM
PERIODS_MM=np.array([16.,32.,64.,128.,256.,512.,1024.])
FEATURE_COUNT=425


def cell_residuals(offsets,positions):
    offsets=np.asarray(offsets,float);positions=np.asarray(positions,float)
    if offsets.shape!=(len(positions),3) or not np.isfinite(offsets).all() or not np.isfinite(positions).all():
        raise ValueError('Three signed offset fields and physical query positions required')
    anchor=np.floor(positions/CELL_MM)*CELL_MM
    return (positions[:,None]+offsets*RADIUS_MM-anchor[:,None])/RADIUS_MM


def cell_offsets(residuals,positions):
    residuals=np.asarray(residuals,float);positions=np.asarray(positions,float)
    if residuals.shape!=(len(positions),3) or not np.isfinite(residuals).all() or not np.isfinite(positions).all():
        raise ValueError('Three bounded center residuals and physical query positions required')
    anchor=np.floor(positions/CELL_MM)*CELL_MM
    return (anchor[:,None]+residuals*RADIUS_MM-positions[:,None])/RADIUS_MM


def phase_features(sequence,indices):
    indices=np.asarray(indices,np.int64);positions=indices.astype(float)*float(sequence['sample_step_mm'])
    coordinates=np.c_[positions,np.asarray(sequence['xyz'],float)[indices]]
    angles=coordinates[:,:,None]*(2.*np.pi/PERIODS_MM)[None,None,:]
    phases=np.mod(positions,CELL_MM)/CELL_MM
    return np.c_[phases,np.sin(angles).reshape(len(indices),-1),np.cos(angles).reshape(len(indices),-1)].astype(np.float32)


def bounded_features(sequence,data,entry,indices):
    values=np.c_[point_features(sequence,data,entry,indices,True),phase_features(sequence,indices)]
    if values.shape!=(len(indices),FEATURE_COUNT) or not np.isfinite(values).all():
        raise ValueError('Finite shared physical phase features required')
    return values.astype(np.float32)
