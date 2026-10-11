"""Complete-source gradient transactions and unchanged source-C budget cursor.

No neural framework, optimizer or dataset is imported. The production caller
supplies the frozen214 per-path latent loss and complete197 certificates.
"""
from dataclasses import dataclass, asdict
import math
import random

VERSION = 'COMPLETE_SOURCE_C_TRANSACTIONS_AND_CURSOR_229_v1'
SOURCE_COUNT = 16
MAX_EPOCHS = 80
MAX_SOURCE_SLOTS = 1280
SEED = 20261002


@dataclass(frozen=True)
class SourceCursor:
    epoch: int = 0
    slot: int = 0
    complete_sources: int = 0
    actual_updates: int = 0

    def __post_init__(self):
        values = (self.epoch, self.slot, self.complete_sources, self.actual_updates)
        if any(isinstance(value,bool) or not isinstance(value,int) for value in values):
            raise ValueError('Integer source-C cursor required')
        if (not 0 <= self.epoch <= MAX_EPOCHS or not 0 <= self.slot < SOURCE_COUNT
                or self.complete_sources != self.epoch*SOURCE_COUNT+self.slot
                or not 0 <= self.actual_updates <= self.complete_sources <= MAX_SOURCE_SLOTS
                or (self.epoch == MAX_EPOCHS and self.slot != 0)):
            raise ValueError('Original80/1280 source-C cursor changed or inconsistent')

    @property
    def done(self):
        return self.complete_sources == MAX_SOURCE_SLOTS

    def source_index(self):
        if self.done:
            raise ValueError('Source-C original budget exhausted')
        order = list(range(SOURCE_COUNT))
        random.Random(SEED+self.epoch).shuffle(order)
        return order[self.slot]

    def advance(self, updated):
        if self.done or not isinstance(updated,bool):
            raise ValueError('One completed source and explicit update flag required')
        count = self.complete_sources+1
        return SourceCursor(count//SOURCE_COUNT,count%SOURCE_COUNT,count,self.actual_updates+int(updated))

    def as_dict(self):
        return asdict(self)


class CompleteSourceFailure(RuntimeError):
    def __init__(self,path,completed_paths,error,*,step_entered=False,step_returned=False,gradients_cleared=True):
        super().__init__(f'Complete source failed at path{path} after{completed_paths} complete paths: {error}')
        self.path=path
        self.complete_paths_before_failure=completed_paths
        self.original_error_type=type(error).__name__
        self.original_error=str(error)
        self.optimizer_step_entered=step_entered
        self.optimizer_step_returned=step_returned
        self.failed_source_gradients_cleared=gradients_cleared


def _certificate(detail, tolerance):
    hashes=[]
    for mode in ('good','bad'):
        value=detail[mode]
        gap=value['mip_gap']
        if (value['status']!='OPTIMAL' or not value['optimal_proved']
                or not value['complete_graph_solution'] or gap is None
                or not math.isfinite(float(gap)) or not 0. <= float(gap) <= tolerance):
            raise ValueError('Full Good/Bad certificate required before any source update')
        hashes.append(value['graph_fingerprint'])
    if not hashes[0] or hashes[0]!=hashes[1]:
        raise ValueError('Complete Good/Bad graph binding differs')


def complete_source_update(views,expected_path_count,*,optimizer,path_loss,backward,gradient_state,
                           record_path,certificate_tolerance=1e-9):
    """No optimizer step until every original path, including empties, succeeds.

    path_loss returns the unchanged event/child normalized214 loss. Division by
    the full original source path count happens exactly once in this function.
    Failed-source gradients are cleared; the caller freezes a failure checkpoint.
    """
    if (isinstance(expected_path_count,bool) or not isinstance(expected_path_count,int)
            or expected_path_count <= 0 or len(views)!=expected_path_count):
        raise ValueError('Complete original source path denominator required')
    if certificate_tolerance!=1e-9:
        raise ValueError('Original214 numerical certificate tolerance required')
    paths=[view[0] for view in views]
    if len(set(paths))!=expected_path_count:
        raise ValueError('Repeated or missing path cannot replace complete source')
    optimizer.zero_grad(set_to_none=True)
    complete,loss_sum,artifacts=0,0.,[]
    current=None
    step_entered,step_returned=False,False
    try:
        for view in views:
            current=view[0]
            loss,detail=path_loss(view)
            _certificate(detail,certificate_tolerance)
            value=float(loss.detach())
            if not math.isfinite(value) or value < 0.:
                raise ValueError('Finite nonnegative normalized identity hinge required')
            backward(loss/expected_path_count)
            artifact=record_path(current,detail)
            artifacts.append(artifact)
            complete+=1
            loss_sum+=value/expected_path_count
            del loss
        gradients,nonzero=gradient_state()
        if not isinstance(nonzero,bool):
            raise ValueError('Explicit finite-gradient update condition required')
        if nonzero:
            step_entered=True
            optimizer.step()
            step_returned=True
        return dict(source_loss=loss_sum,full_path_denominator=expected_path_count,
                    complete_paths=complete,optimizer_step_executed=nonzero,
                    gradient_state=gradients,path_artifacts=artifacts,
                    empty_paths_in_original_denominator=True,version=VERSION)
    except Exception as error:
        cleared=True
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            cleared=False
        raise CompleteSourceFailure(current,complete,error,step_entered=step_entered,
                                    step_returned=step_returned,gradients_cleared=cleared) from error
