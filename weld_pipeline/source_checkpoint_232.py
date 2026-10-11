"""SHA-bound complete-source checkpoints with explicit parameter/RNG devices.

Torch is imported only by actual checkpoint consumers. The commit descriptor
binds both PT and metadata SHA, without a circular embedded metadata hash.
"""
import io
import json
from hashlib import sha256
from pathlib import Path
import random

from .source_c_execution_229 import SourceCursor
from .structural_score_targets_232 import LOSS_KIND
from .supervised_structural_loss_232 import file_sha, object_sha

VERSION = 'COMPLETE_SOURCE_PARAMETER_RNG_DEVICE_CHECKPOINT_232_v2_233_ZERO_BOUNDARY'
BOUNDARY = 'COMPLETE_ORDERED_SOURCE_DEVICE_SYNCHRONIZED_AND_FINITE'
ZERO_BOUNDARY = 'PRE_FORWARD_ZERO_UPDATE_MODEL_BUFFERS_RNG_CURSOR'


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    raise TypeError('Explicit serializable optimizer options required: ' + type(value).__name__)


def tree_hash(value):
    """Hash every tensor/array and scalar, preserving keys, dtype, shape and order."""
    import numpy as np
    import torch
    digest = sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            array = v.detach().cpu().contiguous().numpy()
            digest.update(b'tensor'); digest.update(str(v.dtype).encode())
            digest.update(str(tuple(v.shape)).encode()); digest.update(array.tobytes())
        elif isinstance(v, np.ndarray):
            digest.update(b'array'); digest.update(str(v.dtype).encode())
            digest.update(str(v.shape).encode()); digest.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            digest.update(b'dict')
            for key in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(type(v).__name__.encode()); digest.update(str(len(v)).encode())
            for item in v:
                visit(item)
        elif v is None or isinstance(v, (str, int, float, bool, bytes)):
            digest.update(type(v).__name__.encode()); digest.update(repr(v).encode())
        elif isinstance(v, np.generic):
            digest.update(str(v.dtype).encode()); digest.update(v.tobytes())
        else:
            raise TypeError('Unregistered checkpoint state: ' + type(v).__name__)
    visit(value)
    return digest.hexdigest()


def parameter_binding(model, optimizer):
    names = list(model.named_parameters())
    by_id = {id(p): name for name, p in names}
    if len(by_id) != len(names):
        raise ValueError('Unique named parameter identities required')
    groups = []; observed = []
    for group_index, group in enumerate(optimizer.param_groups):
        ordered = []
        for parameter in group['params']:
            name = by_id.get(id(parameter))
            if name is None or name in observed:
                raise ValueError('Optimizer parameter missing/duplicated/replaced')
            observed.append(name)
            ordered.append(dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                                device=str(parameter.device)))
        groups.append(dict(group_index=group_index, parameters=ordered,
                           actual_options=plain({k: v for k, v in group.items() if k != 'params'})))
    if set(observed) != {name for name, _ in names}:
        raise ValueError('All model parameters must be bound to AdamW groups')
    return dict(optimizer_class=type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
                model_parameter_name_order=[name for name, _ in names], parameter_groups=groups)


def tensor_layout(value):
    import torch
    output = []
    def walk(v, path):
        if isinstance(v, torch.Tensor):
            output.append(dict(path=path, shape=list(v.shape), dtype=str(v.dtype), device=str(v.device)))
        elif isinstance(v, dict):
            for key in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                walk(v[key], path + '/' + type(key).__name__ + ':' + str(key))
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                walk(item, path + '/' + str(i))
    walk(value, 'root')
    return output


def finite_state(model, optimizer):
    import torch
    model_state, optimizer_state = model.state_dict(), optimizer.state_dict()
    count = 0
    def check(v):
        nonlocal count
        if isinstance(v, torch.Tensor):
            count += 1
            if not bool(torch.isfinite(v).all()):
                raise ValueError('Nonfinite model/AdamW checkpoint tensor')
        elif isinstance(v, dict):
            for item in v.values():
                check(item)
        elif isinstance(v, (list, tuple)):
            for item in v:
                check(item)
        elif isinstance(v, float):
            import math
            if not math.isfinite(v):
                raise ValueError('Nonfinite model/AdamW scalar option')
    check(model_state); check(optimizer_state)
    return dict(finite=True, every_model_and_optimizer_tensor_checked=True, tensor_count=count,
                model_content_sha256=tree_hash(model_state), optimizer_content_sha256=tree_hash(optimizer_state),
                parameter_binding=parameter_binding(model, optimizer),
                optimizer_tensor_devices=tensor_layout(optimizer_state))


def gradient_receipt(model):
    import numpy as np
    details = []; nonzero = False
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            details.append(dict(name=name, grad_is_none=True, shape=list(parameter.shape),
                                dtype=str(parameter.dtype), device=str(parameter.device)))
            continue
        array = np.ascontiguousarray(gradient.detach().cpu().numpy())
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite actual accumulated gradient: ' + name)
        changed = bool(np.any(array != 0))
        nonzero = nonzero or changed
        details.append(dict(name=name, grad_is_none=False, shape=list(array.shape),
            dtype=str(array.dtype), device=str(gradient.device), raw_sha256=sha256(array.tobytes()).hexdigest(),
            finite=True, nonzero=changed, maximum_abs=float(np.abs(array).max()) if array.size else 0.))
    return dict(all_parameters_accounted=True, finite=True, nonzero=nonzero, parameters=details,
                condition='ANY_STRICT_NONZERO_VALUE_ACROSS_ALL_ACTUAL_PARAMETER_GRADIENTS')


def synchronize_model(model):
    import torch
    devices = sorted({str(v.device) for v in list(model.parameters()) + list(model.buffers())})
    cuda = [v for v in devices if v.startswith('cuda:')]
    for device in cuda:
        torch.cuda.synchronize(torch.device(device))
    return dict(actual_model_devices=devices, synchronized_cuda_devices=cuda,
                actual_cuda_synchronize_called=bool(cuda), CPU_operations_synchronous=not cuda)


def rng_payload(model, numpy_generators, torch_generators):
    import numpy as np
    import torch
    devices = sorted({str(p.device) for p in model.parameters()})
    cuda = any(v.startswith('cuda:') for v in devices)
    device_map = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            device_map.append(dict(index=index, name=properties.name, total_memory=properties.total_memory,
                                   capability=list(torch.cuda.get_device_capability(index))))
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        CUDA=torch.cuda.get_rng_state_all() if cuda else None, CUDA_device_map=device_map,
        actual_model_devices=devices,
        independent_numpy={name: dict(bit_generator=type(generator.bit_generator).__name__,
                                      state=generator.bit_generator.state)
                           for name, generator in numpy_generators.items()},
        independent_torch={name: dict(device=str(generator.device), state=generator.get_state())
                           for name, generator in torch_generators.items()},
        source_order_generator_policy='FRESH_LOCAL_RANDOM_SEED_PLUS_EPOCH_NO_MUTABLE_INSTANCE')


def commit_checkpoint(directory, label, model, optimizer, cursor, config, *, writer,
                      source_receipt=None, numpy_generators=None, torch_generators=None, extra=None):
    import torch
    numpy_generators = {} if numpy_generators is None else numpy_generators
    torch_generators = {} if torch_generators is None else torch_generators
    if config['loss_kind'] != LOSS_KIND:
        raise ValueError('Registered new method config required')
    cursor = SourceCursor(**cursor.as_dict())
    sync = synchronize_model(model)
    state = finite_state(model, optimizer)
    rng = rng_payload(model, numpy_generators, torch_generators)
    payload = dict(version=VERSION, loss_kind=LOSS_KIND, boundary=BOUNDARY,
        state_dict=model.state_dict(), optimizer=optimizer.state_dict(), cursor=cursor.as_dict(),
        config=config, config_sha256=object_sha(config), parameter_binding=state['parameter_binding'],
        RNG=rng, RNG_content_sha256=tree_hash(rng), model_content_sha256=state['model_content_sha256'],
        optimizer_content_sha256=state['optimizer_content_sha256'],
        optimizer_tensor_devices=state['optimizer_tensor_devices'], extra=extra or {})
    stream = io.BytesIO(); torch.save(payload, stream)
    directory = Path(directory)
    pt = writer.commit_bytes(directory / (label + '.pt'), stream.getvalue())
    metadata = writer.commit_json(directory / (label + '.json'), dict(version=VERSION,
        loss_kind=LOSS_KIND, boundary=BOUNDARY, checkpoint_pt=pt, cursor=cursor.as_dict(),
        config_sha256=payload['config_sha256'], model_content_sha256=state['model_content_sha256'],
        optimizer_content_sha256=state['optimizer_content_sha256'], parameter_binding=state['parameter_binding'],
        optimizer_tensor_devices=state['optimizer_tensor_devices'], RNG_content_sha256=payload['RNG_content_sha256'],
        synchronization=sync, finite=True, source_receipt=source_receipt, full_contract_pass=False))
    return writer.commit_json(directory / (label + '.checkpoint_commit.json'), dict(
        version=VERSION, status='COMPLETE_SOURCE_CHECKPOINT_PAIR_COMMITTED', boundary=BOUNDARY,
        loss_kind=LOSS_KIND, checkpoint_pt=pt, checkpoint_metadata=metadata,
        cursor=cursor.as_dict(), config_sha256=payload['config_sha256'], full_contract_pass=False))


def commit_initial_boundary(directory, model, cursor, config, *, writer):
    """Commit the real zero-update start before any neural forward, without AdamW."""
    import torch
    if cursor != SourceCursor() or config['loss_kind'] != LOSS_KIND:
        raise ValueError('Registered original zero-update cursor required')
    synchronization = synchronize_model(model)
    state = model.state_dict()
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Finite initial parameters and buffers required')
    rng = rng_payload(model, {}, {})
    payload = dict(version=VERSION, boundary=ZERO_BOUNDARY, loss_kind=LOSS_KIND,
        state_dict=state, model_content_sha256=tree_hash(state), model_layout=tensor_layout(state),
        RNG=rng, RNG_content_sha256=tree_hash(rng), cursor=cursor.as_dict(),
        config_sha256=object_sha(config))
    stream = io.BytesIO(); torch.save(payload, stream)
    directory = Path(directory)
    pt = writer.commit_bytes(directory/'initial_zero_boundary.pt', stream.getvalue())
    return writer.commit_json(directory/'initial_zero_boundary.commit.json', dict(
        version=VERSION, status='COMMITTED_PRE_FORWARD_ZERO_UPDATE_BOUNDARY',
        boundary=ZERO_BOUNDARY, loss_kind=LOSS_KIND, checkpoint_pt=pt,
        model_content_sha256=payload['model_content_sha256'], model_layout=payload['model_layout'],
        RNG_content_sha256=payload['RNG_content_sha256'], cursor=cursor.as_dict(),
        config_sha256=object_sha(config), synchronization=synchronization,
        actual_weld_optimizer_updates=0, optimizer_created=False, full_contract_pass=False))


def restore_initial_boundary(path, model, config):
    """Require unchanged model/buffers, then restore the exact pre-forward RNG."""
    descriptor = json.loads(Path(path).read_text('utf-8-sig'))
    if (descriptor['version'] != VERSION or descriptor['boundary'] != ZERO_BOUNDARY
            or descriptor['status'] != 'COMMITTED_PRE_FORWARD_ZERO_UPDATE_BOUNDARY'
            or descriptor['loss_kind'] != LOSS_KIND or descriptor['config_sha256'] != object_sha(config)
            or SourceCursor(**descriptor['cursor']) != SourceCursor()
            or descriptor['actual_weld_optimizer_updates'] != 0 or descriptor['optimizer_created'] is not False):
        raise ValueError('Registered pre-forward zero-update boundary required')
    artifact = descriptor['checkpoint_pt']
    if file_sha(artifact['path']) != artifact['sha256']:
        raise ValueError('Zero-boundary PT SHA differs before load')
    import numpy as np
    import torch
    with Path(artifact['path']).open('rb') as stream:
        saved = torch.load(stream, map_location='cpu')
    for key in ('version', 'boundary', 'loss_kind', 'config_sha256', 'cursor',
                'model_content_sha256', 'model_layout', 'RNG_content_sha256'):
        if saved[key] != descriptor[key]:
            raise ValueError('Zero-boundary descriptor/payload binding differs: ' + key)
    state = model.state_dict()
    synchronize_model(model)
    if (tree_hash(saved['state_dict']) != descriptor['model_content_sha256']
            or tree_hash(state) != descriptor['model_content_sha256']
            or tensor_layout(state) != descriptor['model_layout']
            or tree_hash(saved['RNG']) != descriptor['RNG_content_sha256']):
        raise ValueError('Pre-forward parameters/buffers/device/RNG binding differs')
    rng = saved['RNG']; current = rng_payload(model, {}, {})
    cuda_model = any(device.startswith('cuda:') for device in current['actual_model_devices'])
    if (rng['actual_model_devices'] != current['actual_model_devices']
            or rng['CUDA_device_map'] != current['CUDA_device_map']
            or (rng['CUDA'] is not None) != cuda_model
            or (cuda_model and len(rng['CUDA']) != len(current['CUDA_device_map']))
            or rng['independent_numpy'] or rng['independent_torch']):
        raise ValueError('Zero-boundary actual device or RNG registry differs')
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    if cuda_model:
        torch.cuda.set_rng_state_all(rng['CUDA'])
    if tree_hash(rng_payload(model, {}, {})) != descriptor['RNG_content_sha256']:
        raise ValueError('Actual zero-update RNG restoration differs')
    return dict(status='PASS_PRE_FORWARD_ZERO_UPDATE_MODEL_BUFFERS_AND_RNG_RESTORED',
        descriptor_sha256=file_sha(path), model_buffers_unchanged=True,
        actual_model_devices=rng['actual_model_devices'], actual_cuda_rng_restored=cuda_model,
        RNG_content_sha256=descriptor['RNG_content_sha256'], cursor=descriptor['cursor'],
        actual_weld_optimizer_updates=0, optimizer_created=False, full_contract_pass=False)


def verify_checkpoint_commit(path, config):
    """Verify both committed file hashes before importing Torch or loading PT."""
    descriptor = json.loads(Path(path).read_text('utf-8-sig'))
    if (descriptor['version'] != VERSION or descriptor['status'] != 'COMPLETE_SOURCE_CHECKPOINT_PAIR_COMMITTED'
            or descriptor['boundary'] != BOUNDARY or descriptor['loss_kind'] != LOSS_KIND
            or descriptor['config_sha256'] != object_sha(config)):
        raise ValueError('Committed complete-source descriptor/config required')
    for key in ('checkpoint_pt', 'checkpoint_metadata'):
        artifact = descriptor[key]
        if file_sha(artifact['path']) != artifact['sha256']:
            raise ValueError('Committed PT/metadata SHA differs: ' + key)
    metadata = json.loads(Path(descriptor['checkpoint_metadata']['path']).read_text('utf-8-sig'))
    if (metadata['checkpoint_pt'] != descriptor['checkpoint_pt'] or metadata['cursor'] != descriptor['cursor']
            or metadata['config_sha256'] != object_sha(config) or metadata['finite'] is not True
            or metadata['boundary'] != BOUNDARY or metadata['loss_kind'] != LOSS_KIND):
        raise ValueError('PT and metadata committed relation differs')
    SourceCursor(**descriptor['cursor'])
    return descriptor, metadata


def load_checkpoint(path, model, optimizer, config, *, numpy_generators=None, torch_generators=None):
    """Failure invalidates the passed objects; the caller must abort their use."""
    descriptor, metadata = verify_checkpoint_commit(path, config)
    import numpy as np
    import torch
    numpy_generators = {} if numpy_generators is None else numpy_generators
    torch_generators = {} if torch_generators is None else torch_generators
    with Path(descriptor['checkpoint_pt']['path']).open('rb') as stream:
        saved = torch.load(stream, map_location='cpu')
    expected_binding = parameter_binding(model, optimizer)
    if (saved['version'] != VERSION or saved['boundary'] != BOUNDARY or saved['loss_kind'] != LOSS_KIND
            or saved['config'] != config or saved['config_sha256'] != object_sha(config)
            or saved['cursor'] != descriptor['cursor'] or saved['parameter_binding'] != expected_binding
            or metadata['parameter_binding'] != expected_binding):
        raise ValueError('Parameter names/groups/order/options/device or method binding differs')
    for key, current in (('state_dict', model.state_dict()),):
        if (list(saved[key]) != list(current)
                or [(k, tuple(v.shape), str(v.dtype)) for k, v in saved[key].items()]
                != [(k, tuple(v.shape), str(v.dtype)) for k, v in current.items()]):
            raise ValueError('Model checkpoint schema/name order changed')
    if (tree_hash(saved['state_dict']) != metadata['model_content_sha256']
            or tree_hash(saved['optimizer']) != metadata['optimizer_content_sha256']
            or tree_hash(saved['RNG']) != metadata['RNG_content_sha256']
            or saved['model_content_sha256'] != metadata['model_content_sha256']
            or saved['optimizer_content_sha256'] != metadata['optimizer_content_sha256']
            or saved['RNG_content_sha256'] != metadata['RNG_content_sha256']):
        raise ValueError('Every model/AdamW/RNG content binding must match')
    rng = saved['RNG']
    actual = rng_payload(model, numpy_generators, torch_generators)
    cuda_model = any(v.startswith('cuda:') for v in actual['actual_model_devices'])
    if ((rng['CUDA'] is not None) != cuda_model or rng['actual_model_devices'] != actual['actual_model_devices']
            or rng['CUDA_device_map'] != actual['CUDA_device_map']
            or (cuda_model and len(rng['CUDA']) != len(actual['CUDA_device_map']))
            or set(rng['independent_numpy']) != set(numpy_generators)
            or set(rng['independent_torch']) != set(torch_generators)):
        raise ValueError('Bidirectional CUDA RNG/device mapping or independent RNG registry differs')
    model.load_state_dict(saved['state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    synchronize_model(model)
    restored = finite_state(model, optimizer)
    if (restored['model_content_sha256'] != metadata['model_content_sha256']
            or restored['optimizer_content_sha256'] != metadata['optimizer_content_sha256']
            or restored['parameter_binding'] != expected_binding
            or restored['optimizer_tensor_devices'] != metadata['optimizer_tensor_devices']):
        raise ValueError('Restored actual model/AdamW tensors/order/device differ')
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    if cuda_model:
        torch.cuda.set_rng_state_all(rng['CUDA'])
    for name, generator in numpy_generators.items():
        value = rng['independent_numpy'][name]
        if value['bit_generator'] != type(generator.bit_generator).__name__:
            raise ValueError('Independent NumPy bit generator changed')
        generator.bit_generator.state = value['state']
    for name, generator in torch_generators.items():
        value = rng['independent_torch'][name]
        if value['device'] != str(generator.device):
            raise ValueError('Independent Torch generator device changed')
        generator.set_state(value['state'])
    if tree_hash(rng_payload(model, numpy_generators, torch_generators)) != metadata['RNG_content_sha256']:
        raise ValueError('Actual restored global/independent RNG states differ')
    return SourceCursor(**saved['cursor']), saved['extra'], dict(
        status='PASS_LOADED_COMMITTED_COMPLETE_SOURCE_ALL_CONTENT_PARAMETER_DEVICE_AND_RNG',
        actual_model_devices=rng['actual_model_devices'], actual_cuda_rng_restored=cuda_model,
        descriptor_sha256=file_sha(path), cursor=saved['cursor'], finite=True,
        uncounted_optimizer_updates=0, full_contract_pass=False)
