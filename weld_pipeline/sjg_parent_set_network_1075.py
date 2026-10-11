"""Complete parent-query encoder, exact chunked all-object attention and heads.

No candidate filtering, parent-count oracle, local event summation or NMS.
Endpoint interval inference and set supervision are supplied by the caller.
"""
import math
import torch
from torch import nn


class _ChunkedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, chunk):
        scale = q.shape[-1] ** -0.5
        maximum = q.new_full((*q.shape[:-1],1), -float('inf'))
        normalizer = q.new_zeros((*q.shape[:-1],1))
        numerator = q.new_zeros((*q.shape[:-1],v.shape[-1]))
        for first in range(0,k.shape[-2],chunk):
            logits = (q @ k[:,first:first+chunk].transpose(-1,-2))*scale
            new_maximum = torch.maximum(maximum,logits.amax(-1,keepdim=True))
            previous_scale = torch.exp(maximum-new_maximum)
            probability = torch.exp(logits-new_maximum)
            normalizer = normalizer*previous_scale+probability.sum(-1,keepdim=True)
            numerator = numerator*previous_scale+probability@v[:,first:first+chunk]
            maximum = new_maximum
        output = numerator/normalizer
        log_normalizer = maximum+torch.log(normalizer)
        ctx.save_for_backward(q,k,v,output,log_normalizer)
        ctx.chunk = chunk
        return output

    @staticmethod
    def backward(ctx, gradient):
        q,k,v,output,log_normalizer = ctx.saved_tensors
        scale = q.shape[-1]**-0.5
        dq = torch.zeros_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
        row_dot = (gradient*output).sum(-1,keepdim=True)
        for first in range(0,k.shape[-2],ctx.chunk):
            end = first+ctx.chunk
            kb,vb = k[:,first:end],v[:,first:end]
            probability = torch.exp((q@kb.transpose(-1,-2))*scale-log_normalizer)
            dv[:,first:end] = probability.transpose(-1,-2)@gradient
            ds = probability*(gradient@vb.transpose(-1,-2)-row_dot)
            dq.add_((ds@kb)*scale)
            dk[:,first:end] = (ds.transpose(-1,-2)@q)*scale
        return dq,dk,dv,None


def all_object_attention(q,k,v,chunk=2048):
    if q.ndim!=3 or k.ndim!=3 or v.ndim!=3 or k.shape[-2]!=v.shape[-2] or not k.shape[-2]:
        raise ValueError('Nonempty heads x objects x channels attention required')
    return _ChunkedAttention.apply(q,k,v,int(chunk))


class CrossLayer(nn.Module):
    def __init__(self,width=64,heads=4):
        super().__init__();self.heads=heads;self.width=width
        self.query=nn.Linear(width,width);self.key=nn.Linear(width,width);self.value=nn.Linear(width,width)
        self.out=nn.Linear(width,width);self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))

    def forward(self,query,objects,chunk=2048,query_position=None):
        def heads(value):
            return value.reshape(-1,self.heads,self.width//self.heads).transpose(0,1)
        positioned=query if query_position is None else query+query_position
        value=all_object_attention(heads(self.query(positioned)),heads(self.key(objects)),heads(self.value(objects)),chunk)
        value=value.transpose(0,1).reshape(len(query),self.width)
        query=self.norm1(query+self.out(value))
        return self.norm2(query+self.ff(query))


class ParentSetModel(nn.Module):
    """256 parent queries; each retains up to five independently carried children."""
    def __init__(self,width=64,parents=256,children=5,bins=16,bin_channels=242):
        super().__init__()
        self.width=width;self.parents=parents;self.child_slots=children;self.bins=bins;self.bin_channels=bin_channels
        self.context=nn.Linear(669,width)
        self.bin_encoder=nn.Linear(bins*bin_channels,width)
        self.object_norm=nn.LayerNorm(width)
        self.parent_query=nn.Parameter(torch.randn(parents,width)*0.02)
        self.child_position=nn.Parameter(torch.randn(children,width)*0.02)
        self.cross=nn.ModuleList([CrossLayer(width,4),CrossLayer(width,4)])
        self.exists=nn.Linear(width,1)
        self.count=nn.Linear(width,children)
        self.child_state=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.LayerNorm(width))
        self.profile_query=nn.Linear(width,width)
        self.profile_key=nn.Linear(width,width)
        self.weld_type=nn.Linear(width,2)
        self.direction=nn.Linear(width,1)
        self.closed=nn.Linear(width,1)
        self.wrap=nn.Linear(width,1)
        self.parent_closed=nn.Linear(width,1)
        self.connection=nn.Linear(width,3)  # unknown, shared, positive gap
        self.side=nn.Linear(width,3)  # unknown, side A, side B; unknown is retained
        self.endpoint_query=nn.Linear(width,width*2)
        self.endpoint_key=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width))
        self.fraction=nn.Linear(width,2)

    def encode_objects(self,context,ordered_bins):
        if context.shape[-1]!=669 or ordered_bins.shape[-2:]!=(self.bins,self.bin_channels):
            raise ValueError('Complete registered context and ordered bins required')
        return self.object_norm(self.context(context)+self.bin_encoder(ordered_bins.flatten(-2)))

    def decode_queries(self,objects,chunk=2048):
        query=self.parent_query
        for layer in self.cross:query=layer(query,objects,chunk)
        return query

    def forward_encoded(self,objects,chunk=2048):
        if len(objects)>200000:raise RuntimeError('Object budget exceeded; no truncation')
        query=self.decode_queries(objects,chunk)
        child=self.child_state(query[:,None]+self.child_position[None])
        return dict(existence=self.exists(query).squeeze(-1),count=self.count(query),parent_closed=self.parent_closed(query).squeeze(-1),
            profile_query=self.profile_query(child),profile_key=self.profile_key(objects),
            type=self.weld_type(child),direction=self.direction(child).squeeze(-1),
            closed=self.closed(child).squeeze(-1),wrap=self.wrap(child).squeeze(-1),connection=self.connection(child),
            side=self.side(child),endpoint_query=self.endpoint_query(child).reshape(self.parents,self.child_slots,2,self.width),
            fraction=self.fraction(child).sigmoid(),parent_latent=query,child_latent=child)

    def forward(self,context,ordered_bins,chunk=2048):
        return self.forward_encoded(self.encode_objects(context,ordered_bins),chunk)

    def endpoint_logits(self,query,interval_features):
        """All original anchor intervals participate; caller supplies no GT window."""
        return query@self.endpoint_key(interval_features).transpose(-1,-2)/math.sqrt(self.width)


def attention_controls():
    torch.manual_seed(1031)
    report=[]
    for dtype,tolerance in ((torch.float64,1e-10),(torch.float32,2e-5)):
        original=[torch.randn(shape,dtype=dtype) for shape in ((4,7,16),(4,53,16),(4,53,16))]
        q,k,v=[x.clone().requires_grad_() for x in original]
        dense=((q@k.transpose(-1,-2))/4).softmax(-1)@v
        upstream=torch.randn_like(dense)
        reference=torch.autograd.grad((dense*upstream).sum(),(q,k,v))
        for size in (1,11,53,100):
            sq,sk,sv=[x.clone().requires_grad_() for x in original]
            actual=all_object_attention(sq,sk,sv,size)
            gradients=torch.autograd.grad((actual*upstream).sum(),(sq,sk,sv))
            error=float((actual-dense).abs().max())
            gradient_error=max(float((a-b).abs().max()) for a,b in zip(gradients,reference))
            assert error<tolerance and gradient_error<tolerance,(dtype,size,error,gradient_error)
            report.append(dict(dtype=str(dtype),chunk=size,forward_max_error=error,gradient_max_error=gradient_error))
    # All object identities remain in the denominator, including duplicate geometry.
    q=torch.tensor([[[1.]]]);k=torch.tensor([[[0.],[0.],[0.]]]);v=torch.tensor([[[0.],[0.],[3.]]])
    assert torch.allclose(all_object_attention(q,k,v,1),torch.ones_like(q))
    return dict(dense_forward_and_gradients=report,all_objects_in_softmax=True,optimizer_updates=0)
