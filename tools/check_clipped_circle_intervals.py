"""Numerical controls for annular clipping; no workpiece labels."""
import numpy as np
from weld_pipeline.server_clipped_graph_circle_intervals import clipped_supported_circle_intervals as clip
from weld_pipeline.server_graph_circle_intervals import supported_circle_intervals as whole


def covered(intervals, theta):
    result=np.zeros(len(theta), bool)
    for a,b in intervals:
        result |= ((theta>=a)&(theta<=b))|((theta+2*np.pi>=a)&(theta+2*np.pi<=b))
    return result


def main():
    rng=np.random.default_rng(1031)
    vertices=rng.uniform(-15,15,(400,2)); edges=np.arange(400).reshape(-1,2)
    angles=np.linspace(0,2*np.pi,20001,endpoint=False)
    result=clip(vertices,edges,[0,0],10,1)
    assert np.array_equal(covered(result,angles),covered(clip(vertices,edges[::-1,::-1],[0,0],10,1),angles))
    assert np.all(~covered(whole(vertices,edges,[0,0],10,1),angles)|covered(result,angles))
    # Independent dense segment sampling: all strict-interior observed samples
    # must project into support. Leave radial edges (zero angular measure) out.
    t=np.linspace(0,1,1001)
    points=vertices[edges[:,0],None,:]*(1-t[None,:,None])+vertices[edges[:,1],None,:]*t[None,:,None]
    radii=np.linalg.norm(points,axis=2); inside=(radii>9+1e-8)&(radii<11-1e-8)
    theta=np.mod(np.arctan2(points[:,:,1],points[:,:,0]),2*np.pi)
    assert covered(result,theta[inside]).all()
    # An edge through the forbidden inner disk must retain two separated pieces.
    v=np.array([[-12.,3.],[12.,3.]])
    intervals=clip(v,np.array([[0,1]]),[0,0],10,1)
    assert len(intervals)==2
    assert not covered(intervals,np.array([np.pi/2]))[0]
    # Two observed disconnected edges cannot fill the intervening gap.
    theta=np.array([0.,.1,.3,.4]); v=10*np.column_stack((np.cos(theta),np.sin(theta)))
    intervals=clip(v,np.array([[0,1],[2,3]]),[0,0],10,1)
    assert len(intervals)==2 and not covered(intervals,np.array([.2]))[0]
    assert clip(np.array([[30.,0.],[31.,1.]]),np.array([[0,1]]),[0,0],10,1)==[]
    # A partially supported edge gains its observed annulus portion only.
    v=np.array([[10.,0.],[10.,8.]])
    assert whole(v,np.array([[0,1]]),[0,0],10,1)==[]
    assert len(clip(v,np.array([[0,1]]),[0,0],10,1))==1
    print('PASS reversal, whole-edge containment, dense observed support, inner-disk exclusion, positive-gap preservation, remote edge, partial edge')

if __name__=='__main__':main()
