"""Exact frozen-input encoder with block recomputation instead of retaining GBs."""
import numpy as np
import torch
import torch.nn.functional as F


def _release_input_pages(device,first):
    if device.type=='cuda' and first%8192==0:
        from weld_pipeline.sjg_local_runtime_904 import resources,reclaim_own_idle_pages,check_resources
        if resources()['available_physical_bytes']<1879048192:
            reclaim_own_idle_pages();check_resources()


def _encode(context,bins,parameters,eps):
    cw,cb,bw,bb,nw,nb=parameters
    value=F.linear(context,cw,cb)+F.linear(bins.flatten(-2),bw,bb)
    return F.layer_norm(value,(len(nb),),nw,nb,eps)


class _StreamEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx,data,chunk,eps,*parameters):
        ctx.data=data;ctx.chunk=int(chunk);ctx.eps=eps;ctx.save_for_backward(*parameters)
        device=parameters[0].device
        output=parameters[0].new_empty((len(data),len(parameters[-1])))
        for first in range(0,len(data),ctx.chunk):
            _release_input_pages(device,first)
            c,b=data.arrays(slice(first,first+ctx.chunk))
            c=torch.as_tensor(c,device=device);b=torch.as_tensor(b,device=device)
            output[first:first+len(c)]=_encode(c,b,parameters,eps)
        return output

    @staticmethod
    def backward(ctx,gradient):
        original=ctx.saved_tensors
        parameters=tuple(p.detach().requires_grad_() for p in original)
        accumulated=[torch.zeros_like(p) for p in parameters]
        for first in range(0,len(ctx.data),ctx.chunk):
            _release_input_pages(gradient.device,first)
            c,b=ctx.data.arrays(slice(first,first+ctx.chunk))
            c=torch.as_tensor(c,device=gradient.device);b=torch.as_tensor(b,device=gradient.device)
            with torch.enable_grad():
                output=_encode(c,b,parameters,ctx.eps)
                values=torch.autograd.grad(output,parameters,gradient[first:first+len(c)])
            for total,value in zip(accumulated,values):total.add_(value)
        return (None,None,None,*accumulated)


def stream_encode(model,data,chunk=1024):
    """Every original object encoded, exact same function and parameter gradients."""
    return _StreamEncode.apply(data,chunk,model.object_norm.eps,
        model.context.weight,model.context.bias,model.bin_encoder.weight,model.bin_encoder.bias,
        model.object_norm.weight,model.object_norm.bias)


def controls():
    from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel
    torch.manual_seed(1031)
    model=ParentSetModel(bins=2,bin_channels=3).double()
    c=torch.randn(19,669,dtype=torch.float64);b=torch.randn(19,2,3,dtype=torch.float64)
    class Data:
        def __len__(self):return len(c)
        def arrays(self,indices):return c[indices].numpy(),b[indices].numpy()
    parameters=(model.context.weight,model.context.bias,model.bin_encoder.weight,model.bin_encoder.bias,
                model.object_norm.weight,model.object_norm.bias)
    dense=model.encode_objects(c,b);upstream=torch.randn_like(dense)
    expected=torch.autograd.grad(dense,parameters,upstream)
    rows=[]
    for chunk in (1,7,100):
        actual=stream_encode(model,Data(),chunk)
        gradients=torch.autograd.grad(actual,parameters,upstream)
        error=max(float((a-e).abs().max()) for a,e in zip(gradients,expected))
        assert torch.allclose(actual,dense,atol=1e-12,rtol=1e-12) and error<1e-10
        rows.append(dict(chunk=chunk,gradient_max_error=error))
    return dict(all_objects_retained=True,dense_gradient_comparison=rows,optimizer_updates=0)
