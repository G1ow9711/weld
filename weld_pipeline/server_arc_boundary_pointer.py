"""Variable-length ordered boundary selection over frozen input events."""
import torch
from torch import nn


class ArcBoundaryPointer(nn.Module):
    def __init__(self, input_dim=76, hidden=128):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.initial=nn.Linear(2*hidden,hidden)
        self.start=nn.Parameter(torch.zeros(hidden))
        self.recurrent=nn.GRUCell(hidden,hidden)
        self.key=nn.Linear(hidden,hidden,bias=False)
        self.query=nn.Linear(hidden,hidden,bias=False)
        self.energy=nn.Linear(hidden,1,bias=False)
        self.stop=nn.Linear(hidden,1)

    def prepare(self,features):
        embedded=self.encoder(features)
        context=torch.cat((embedded.mean(0),embedded.max(0).values))
        return embedded,torch.tanh(self.initial(context))

    def step(self,embedded,state,previous,positions):
        previous_embedding=self.start if previous<0 else embedded[previous]
        state=self.recurrent(previous_embedding[None],state[None])[0]
        scores=self.energy(torch.tanh(self.key(embedded)+self.query(state)[None])).squeeze(-1)
        # Coordinates, not index order, define strict monotonicity.
        if previous>=0:scores=scores.masked_fill(positions<=positions[previous],-torch.inf)
        return torch.cat((scores,self.stop(state).reshape(1))),state

    def set_teacher_loss(self,features,positions,positive_sets,teacher_path):
        if len(positive_sets)!=len(teacher_path):raise ValueError('Boundary label/path mismatch')
        if any(not positions[b]>positions[a] for a,b in zip(teacher_path,teacher_path[1:])):raise ValueError('Teacher path must strictly increase')
        embedded,state=self.prepare(features);previous=-1;losses=[]
        for positives,chosen in zip(positive_sets,teacher_path):
            scores,state=self.step(embedded,state,previous,positions)
            positives=torch.as_tensor(positives,device=scores.device,dtype=torch.long)
            if previous>=0:positives=positives[positions[positives]>positions[previous]]
            if not len(positives) or not bool((positives==chosen).any()):raise ValueError('Missing or inconsistent supervised boundary')
            losses.append(torch.logsumexp(scores,0)-torch.logsumexp(scores[positives],0))
            previous=chosen
        scores,_=self.step(embedded,state,previous,positions)
        losses.append(torch.logsumexp(scores,0)-scores[-1])
        return torch.stack(losses).mean()

    @torch.no_grad()
    def decode(self,features,positions):
        embedded,state=self.prepare(features);previous=-1;path=[]
        # Each non-STOP action advances to a new event, so N+1 is exhaustive.
        for _ in range(len(positions)+1):
            scores,state=self.step(embedded,state,previous,positions);chosen=int(scores.argmax())
            if chosen==len(positions):return path
            path.append(chosen);previous=chosen
        raise RuntimeError('Ordered pointer did not terminate')
