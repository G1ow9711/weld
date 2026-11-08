"""Restore literal single-primitive input identity before selection and export."""
from .curveonly_01a1ac7c_source_endpoint_v203 import *
from . import curveonly_01a1ac7c_source_endpoint_v203 as previous
from . import curveonly_01a1ac7c_source_selection_v56 as provenance


def normalize(proposals,packet):
    lengths=[float(arclength(c['points'])[-1]) for c in packet['candidates']]
    output,records=[],[]
    tolerance=codec.reference.PROJECTION_NUMERIC_MM
    for proposal in proposals:
        ci=proposal['candidate'];interval=proposal['interval']
        parts=provenance.primitive_intervals(ci,interval['start_mm'],interval['end_mm'],packet,lengths)
        if len(parts)!=1 or parts[0][0]==ci:
            output.append(proposal)
            continue
        primitive,a,b=parts[0]
        # A complete literal segment only: do not strip a measured clearance,
        # split a multijoint span, extend a range, or infer identity by proximity.
        if abs((b-a)-(interval['end_mm']-interval['start_mm']))>tolerance:
            output.append(proposal)
            continue
        points=codec.geometry.native_slice(packet['candidates'][primitive]['points'],a,b)
        record=dict(original_candidate=ci,original_interval=interval,
            primitive_candidate=primitive,primitive_interval_mm=[a,b],
            range_length_delta_mm=abs((b-a)-(interval['end_mm']-interval['start_mm'])),
            source='IDENTICAL_INPUT_PRIMITIVE_PARAMETER_RANGE',new_joint_or_process_decision=False)
        output.append(dict(proposal,candidate=primitive,points=points.tolist(),
            interval=dict(interval,start_mm=a,end_mm=b),finite_representation_provenance=record))
        records.append(record)
    return output,dict(input_proposals=len(proposals),output_proposals=len(output),
        normalized_single_primitive_representations=records,original_geometry_and_true_gaps_not_inferred=True,
        no_labels_or_workpiece_names_or_new_thresholds=True)


class PairedBoundarySet(previous.PairedBoundarySet):
    def predict(self,features,packet):
        proposals=super().predict(features,packet)
        result,self.last_representation_normalization=normalize(proposals,packet)
        self.last_timing=dict(self.last_timing,primitive_output_normalization=self.last_representation_normalization)
        return result
