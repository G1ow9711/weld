"""Matched pointer variant: feed model-preferred valid boundary events back."""
import torch
from .server_arc_boundary_pointer import ArcBoundaryPointer


class ArcBoundaryFeedbackPointer(ArcBoundaryPointer):
    def __init__(self, feedback_mode='fixed', **kwargs):
        super().__init__(**kwargs)
        if feedback_mode not in ('fixed','model_valid'):raise ValueError(feedback_mode)
        self.feedback_mode=feedback_mode

    def set_teacher_loss(self,features,positions,positive_sets,teacher_path):
        if self.feedback_mode=='fixed':
            return super().set_teacher_loss(features,positions,positive_sets,teacher_path)
        if len(positive_sets)!=len(teacher_path):raise ValueError('Boundary label/path mismatch')
        # Backward reachability preserves a strictly increasing completion even
        # where neighboring acceptable endpoint windows overlap.
        feasible=[];ceiling=None
        for values in reversed(positive_sets):
            ids=torch.as_tensor(values,device=positions.device,dtype=torch.long)
            if ceiling is not None:ids=ids[positions[ids]<ceiling]
            if not len(ids):raise ValueError('No ordered completion through label sets')
            feasible.append(ids);ceiling=positions[ids].max()
        feasible.reverse();embedded,state=self.prepare(features);previous=-1;losses=[]
        for ids in feasible:
            scores,state=self.step(embedded,state,previous,positions)
            if previous>=0:ids=ids[positions[ids]>positions[previous]]
            if not len(ids):raise ValueError('Lost reachable boundary set')
            losses.append(torch.logsumexp(scores,0)-torch.logsumexp(scores[ids],0))
            previous=int(ids[scores[ids].detach().argmax()])
        scores,_=self.step(embedded,state,previous,positions)
        losses.append(torch.logsumexp(scores,0)-scores[-1])
        return torch.stack(losses).mean()
