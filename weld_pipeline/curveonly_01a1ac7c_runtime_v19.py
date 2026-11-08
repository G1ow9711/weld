"""Reject an uncertified STEP pose and use the existing raw-XYZ curve path."""
import time,types
from .curveonly_01a1ac7c_runtime_v18 import *
from . import curveonly_01a1ac7c as original
from . import curveonly_01a1ac7c_registration_v2 as registration
from . import curveonly_01a1ac7c_sampler_v5 as sampler
from . import curveonly_01a1ac7c_projected_v11 as projected


class UncertifiedPose(Exception):
    def __init__(self,audit):self.audit=audit


def input_curves(record):
    start=time.perf_counter();sampler.reset_audit();traces=[];raw_cache={};read_seconds=0.
    def read_xyz(path):
        nonlocal read_seconds
        if path not in raw_cache:
            tick=time.perf_counter();raw_cache[path]=original.read_xyz(path);read_seconds+=time.perf_counter()-tick
        return raw_cache[path]
    def choose_pose(*args,**kwargs):
        transform,audit=registration.choose_pose(*args,**kwargs)
        if not audit.get('passed',False):raise UncertifiedPose(audit)
        return transform,audit
    def point_curves(xyz):
        curves,audit=projected.proposals(xyz);traces.append(audit);return curves
    function=types.FunctionType(original.input_curves.__code__,dict(original.__dict__,
        read_xyz=read_xyz,choose_pose=choose_pose,curve_points=sampler.curve_points,point_curves=point_curves))
    rejected=None;reject_elapsed=0.
    try:features,packet=function(record)
    except UncertifiedPose as exc:
        rejected=exc.audit;reject_elapsed=time.perf_counter()-start
        features,packet=function(dict(record,STEP=None))
    packet['native_sampling']=sampler.audit();packet['point_trace_proposals']=traces
    packet['STEP_pose_gate']=dict(required_passed_flag=True,pose_threshold_unchanged=True,
        rejected=rejected is not None,action='XYZ_ONLY'if rejected is not None else'EXISTING_PATH',
        rejected_pose_audit=rejected)
    if rejected is not None:
        packet['timing']['rejected_STEP_attempt_including_raw_read']=reject_elapsed
        packet['timing']['read_xyz']=read_seconds
        packet['timing']['input_to_features']=time.perf_counter()-start
        packet['source']='RAW_XYZ_AFTER_UNCERTIFIED_STEP_POSE'
    return features,packet
