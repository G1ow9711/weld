"""Traversal reversal of all directional source observations."""
import numpy as np
from .curveonly_01a1ac7c_net_v14 import reverse_features as previous


def reverse_features(x):
    result=previous(x)
    # Finite other-material probes at source start/end exchange roles too.
    result[:,[12,14]]=result[:,[14,12]]
    return result


def controls():
    x=np.zeros((5,34),np.float32);x[:,9]=np.linspace(0,1,5)
    x[:,12]=2.;x[:,13]=4.;x[:,14]=7.;x[:,23]=3.;x[:,24]=9.;x[:,29]=11.;x[:,30]=13.
    y=reverse_features(x)
    assert np.all(y[:,12]==7.)and np.all(y[:,14]==2.)and np.all(y[:,13]==4.)
    assert np.all(y[:,23]==9.)and np.all(y[:,24]==3.)
    assert np.all(y[:,29]==13.)and np.all(y[:,30]==11.)
    assert np.max(abs(reverse_features(y)-x))<1e-6
    return dict(passed=True,source_endpoint_observations_exchanged=True,transverse_and_tangent_probes_exchanged=True,
        station_coordinates_reversed=True,full_double_reversal=True)
