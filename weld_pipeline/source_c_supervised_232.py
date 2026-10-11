"""Full ordered-source transactions for the separately registered232 loss."""
import math

from .source_c_execution_229 import SourceCursor
from .structural_score_targets_232 import LOSS_KIND
from .supervised_structural_loss_232 import object_sha

VERSION = 'ORDERED_COMPLETE_SOURCE_C_SUPERVISED_TRANSACTIONS_232_v2_233_GUARDS'
ZERO_GRADIENT_POLICY = 'SKIP_ADAMW_IF_ALL_ACTUAL_PARAMETER_GRADIENTS_STRICTLY_ZERO'


class SupervisedSourceFailure(RuntimeError):
    def __init__(self, path, complete, error, *, entered, returned, synchronized, finite, cleared):
        super().__init__(f'Complete source failed at path {path} after {complete} paths: {error}')
        self.path_id = path
        self.completed_paths = complete
        self.original_error_type = type(error).__name__
        self.original_error = str(error)
        self.step_entered = entered
        self.step_returned_on_host = returned
        self.device_update_synchronized = synchronized
        self.post_update_finite = finite
        self.failed_source_gradients_cleared = cleared


def validate_actual_scalar_loss(loss, detail):
    """Bind the recorded value to the connected scalar actually backpropagated."""
    import torch
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
        raise ValueError('Connected scalar neural loss required')
    measured = float(loss.detach().item())
    if not math.isfinite(measured) or measured < 0 or measured != float(detail['path_loss']):
        raise ValueError('Recorded loss differs from actual backward loss')
    return measured


def complete_supervised_source(views, expected_scopes, *, manifest_sha256,
        optimizer, path_loss, validate_loss, backward, gradient_receipt, record_path,
        synchronize, post_update_validate, model_hash):
    """All paths precede one AdamW step; a skipped step still uses one source slot."""
    expected_paths = tuple(scope['path_id'] for scope in expected_scopes)
    actual_paths = tuple(view[0] for view in views)
    if (not expected_paths or len(set(expected_paths)) != len(expected_paths)
            or actual_paths != expected_paths
            or any(scope['manifest_sha256'] != manifest_sha256 for scope in expected_scopes)
            or any(scope['source_path_denominator'] != len(expected_paths) for scope in expected_scopes)
            or any(object_sha({k: v for k, v in scope.items() if k != 'target_scope_sha256'})
                   != scope['target_scope_sha256'] for scope in expected_scopes)):
        raise ValueError('Exact complete ordered source manifest required before any backward/update')
    optimizer.zero_grad(set_to_none=True)
    before = model_hash()
    complete, source_loss, artifacts, current = 0, 0., [], None
    entered = returned = synchronized = finite = False
    try:
        for view, scope in zip(views, expected_scopes):
            current = view[0]
            loss, detail = path_loss(view)
            if (detail.get('loss_kind') != LOSS_KIND or detail.get('path_id') != current
                    or detail.get('target_scope_sha256') != scope['target_scope_sha256']
                    or detail.get('manifest_sha256') != manifest_sha256
                    or detail.get('training_native_solver_calls') != 0
                    or detail.get('source_denominator_applied_here') is not False
                    or detail.get('analytic_empty') != scope['analytic_empty']):
                raise ValueError('Current complete-path target/method binding differs')
            value = detail['path_loss']
            if not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative actual path loss required')
            if scope['analytic_empty']:
                if (loss is not None or value != 0 or detail.get('actual_neural_forward') is not False
                        or detail.get('analytic_proof_kind') != 'EMPTY_REGISTERED_SCORE_SUPERVISION'):
                    raise ValueError('Only proved empty supervision has zero/no-backward contribution')
            else:
                if loss is None or detail.get('actual_neural_forward') is not True:
                    raise ValueError('Actual complete-candidate neural loss required')
                measured = validate_loss(loss, detail)
                if not math.isfinite(measured) or measured < 0 or measured != value:
                    raise ValueError('Recorded loss differs from actual backward loss')
                backward(loss / len(expected_paths))
            artifacts.append(record_path(current, detail))
            complete += 1
            source_loss += value / len(expected_paths)
            del loss
        backward_sync = synchronize()
        gradients = gradient_receipt()
        if (gradients.get('all_parameters_accounted') is not True or gradients.get('finite') is not True
                or not isinstance(gradients.get('nonzero'), bool)):
            raise ValueError('Every actual parameter gradient must be checked')
        if model_hash() != before:
            raise ValueError('Forward/backward changed model state before the source update')
        updated = gradients['nonzero']
        if updated:
            entered = True
            optimizer.step()
            returned = True
        update_sync = synchronize()
        synchronized = True
        post = post_update_validate()
        if post.get('finite') is not True:
            raise ValueError('Updated model and AdamW state must be finite')
        finite = True
        if not updated and model_hash() != before:
            raise ValueError('Strict all-zero source unexpectedly changed model state')
        return dict(version=VERSION, loss_kind=LOSS_KIND, source_loss=source_loss,
            complete_paths=complete, full_path_denominator=len(expected_paths),
            full_ordered_path_ids=list(expected_paths), manifest_sha256=manifest_sha256,
            model_state_hash_before=before, model_state_hash_after=model_hash(),
            optimizer_step_executed=updated, step_entered=entered, step_returned_on_host=returned,
            device_update_synchronized=synchronized, post_update_finite=finite,
            backward_synchronization=backward_sync, update_synchronization=update_sync,
            zero_gradient_policy=ZERO_GRADIENT_POLICY, gradient_receipt=gradients,
            post_update_receipt=post, path_artifacts=artifacts,
            analytic_empty_paths_still_in_original_denominator=True,
            training_native_solver_calls=0, full_contract_pass=False)
    except Exception as error:
        cleared = True
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            cleared = False
        raise SupervisedSourceFailure(current, complete, error, entered=entered, returned=returned,
                                      synchronized=synchronized, finite=finite, cleared=cleared) from error
