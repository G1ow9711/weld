"""Pro363: best legal versus best illegal action, no candidate-count dilution."""
import numpy as np


def pairwise_oracle_loss(logits,available,allowed):
    scores=np.asarray(logits,dtype=np.float64)
    available=np.asarray(available,dtype=bool);allowed=np.asarray(allowed,dtype=bool)
    if scores.ndim!=1 or available.shape!=scores.shape or allowed.shape!=scores.shape:
        raise ValueError('Action score/mask shape mismatch')
    if not np.isfinite(scores).all() or not allowed.any() or np.any(allowed&~available):
        raise ValueError('Invalid action scores or oracle mask')
    good=np.flatnonzero(allowed);bad=np.flatnonzero(available&~allowed)
    gradient=np.zeros_like(scores)
    if not len(bad):return 0.,gradient
    positive=int(good[np.argmax(scores[good])]);negative=int(bad[np.argmax(scores[bad])])
    difference=float(scores[negative]-scores[positive])
    if not np.isfinite(difference):raise FloatingPointError('Nonfinite pairwise difference')
    if difference>=0:derivative=1./(1.+np.exp(-difference))
    else:
        exp=np.exp(difference);derivative=exp/(1.+exp)
    gradient[negative]=derivative;gradient[positive]=-derivative
    return float(np.logaddexp(0.,difference)),gradient


def check_frozen_state(scores,available,allowed):
    """Numerical differentiation of the selected active branch, including ties."""
    loss,gradient=pairwise_oracle_loss(scores,available,allowed)
    good=np.flatnonzero(allowed);bad=np.flatnonzero(available&~allowed)
    assert len(bad)
    positive=int(good[np.argmax(scores[good])]);negative=int(bad[np.argmax(scores[bad])])
    assert gradient[positive]<0 and gradient[negative]>0
    mask=np.ones(len(scores),bool);mask[[positive,negative]]=False
    assert np.all(gradient[mask]==0)
    errors=[]
    for index in (positive,negative):
        plus=scores.copy();minus=scores.copy();plus[index]+=1e-6;minus[index]-=1e-6
        numerical=(np.logaddexp(0.,plus[negative]-plus[positive])-np.logaddexp(0.,minus[negative]-minus[positive]))/2e-6
        errors.append(float(abs(numerical-gradient[index])))
    assert max(errors)<1e-8
    if scores[negative]>=scores[positive]:assert loss>=np.log(2.)
    updated=scores-.001*gradient
    assert updated[positive]>scores[positive] and updated[negative]<scores[negative]
    assert pairwise_oracle_loss(updated,available,allowed)[0]<=loss+1e-12
    return dict(loss=loss,best_legal=positive,best_illegal=negative,
        legal_logit=float(scores[positive]),illegal_logit=float(scores[negative]),
        legal_gradient=float(gradient[positive]),illegal_gradient=float(gradient[negative]),
        active_branch_maximum_gradient_error=max(errors),other_actions_zero_gradient=True,
        gradient_direction_correct=True,tie_subgradient_uses_stable_original_order=True)
