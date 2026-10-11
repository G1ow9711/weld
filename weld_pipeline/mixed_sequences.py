"""Shared point-only generator for straight and circular weld candidates."""
import numpy as np
from weld_pipeline.bounded_sequences import make_sequences
from weld_pipeline.surface_features import append_surface_features
from weld_pipeline.curved_sequences import make_arc_sequences,arc_training_targets
from weld_pipeline.arc_tangent_sequences import make_tangent_sequences
from weld_pipeline.typed_sequences import training_targets

FEATURE_VERSION='typed_contact_mixed_surface_v1'


def append_curved_contacts(xyz,straight):
    arcs,arc_stats=make_arc_sequences(xyz);tangents,tangent_stats=make_tangent_sequences(xyz)
    accepted=[];duplicates=0
    for candidate in tangents:
        q=candidate['xyz'][[0,-1]].astype(float);duplicate=False
        for base in straight:
            if base['type']!=candidate['type']:continue
            u=base['direction'];p=base['xyz'];position=(q-p[0])@u
            error=np.linalg.norm(np.cross(q-p[0],u),axis=1).max()
            if error<=.25 and position.min()>=-.25 and position.max()<=(len(p)-1)*base['sample_step_mm']+.25:
                duplicate=True;break
        if duplicate:duplicates+=1
        else:accepted.append(candidate)
    return [*straight,*accepted,*arcs],{'arcs':arc_stats,'tangents':tangent_stats,
             'duplicate_tangents_removed':duplicates,'added_tangents':len(accepted)}


def make_mixed_sequences(xyz):
    straight,geometry=make_sequences(xyz);straight,surface=append_surface_features(xyz,straight)
    sequences,curves=append_curved_contacts(xyz,straight)
    return sequences,{'straight_geometry':geometry,'surface':surface,'curves':curves,'feature_version':FEATURE_VERSION}


def mixed_training_targets(sequences,truth):
    line_indices=[i for i,s in enumerate(sequences) if s.get('curve_kind')!='arc']
    arc_indices=[i for i,s in enumerate(sequences) if s.get('curve_kind')=='arc']
    lines=[w for w in truth if all(s['kind']=='line' for s in w['segments'])]
    arcs=[w for w in truth if all(s['kind']=='arc' for s in w['segments'])]
    if len(lines)+len(arcs)!=len(truth):raise ValueError('Mixed primitives within one weld require a joined-path representation')
    y,lm=training_targets([sequences[i] for i in line_indices],lines)
    ay,am=arc_training_targets([sequences[i] for i in arc_indices],arcs)
    targets=[None]*len(sequences)
    for i,t in zip(line_indices,y):targets[i]=t
    for i,t in zip(arc_indices,ay):targets[i]=t
    return targets,{'total_reference_welds':len(truth),
                    'mapped_to_sequences':lm['mapped_to_sequences']+am['mapped_to_sequences'],
                    'line_mapping':lm,'arc_mapping':am}
