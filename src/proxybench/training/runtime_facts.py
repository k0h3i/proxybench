"""Read the initialized training stack without changing numerical settings."""

import os
import subprocess


def optimizer_identity(model, optimizer):
    import torch
    from torch.optim.optimizer import _default_to_fused_or_foreach
    names = {id(value): name for name, value in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        parameters = group['params']
        fused, foreach = group.get('fused'), group.get('foreach')
        if fused is None and foreach is None:
            _, foreach = _default_to_fused_or_foreach(
                parameters, group.get('differentiable', False), use_fused=False)
            if foreach and isinstance(group['lr'], torch.Tensor) and not group.get('capturable', False):
                foreach = False
        groups.append(dict(
            parameters=[names[id(value)] for value in parameters],
            dtypes=sorted({str(value.dtype) for value in parameters}),
            settings={key: value for key, value in group.items() if key != 'params'},
            resolved_backend='fused' if fused else 'foreach' if foreach else 'single_tensor'))
    return dict(class_name=f'{type(optimizer).__module__}.{type(optimizer).__qualname__}',
                backend_evidence='installed Torch dispatch resolver with current parameter types', groups=groups)


def device_and_kernels(torch):
    """Call only after the bounded worker initializes CUDA."""
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5, check=True)
        drivers = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError):
        drivers = ['unknown']
    hardware = dict(name=properties.name, capability=list(torch.cuda.get_device_capability()),
                    total_memory=properties.total_memory, device_uuid=str(getattr(properties, 'uuid', 'unknown')),
                    drivers=drivers, cuda_runtime=torch.version.cuda,
                    cudnn_version=torch.backends.cudnn.version())
    kernels = dict(
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        gradient_checkpointing='unsloth', use_cache=False,
        environment={name: os.environ.get(name, 'unset') for name in (
            'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF',
            'PYTORCH_ALLOC_CONF', 'UNSLOTH_COMPILE_DISABLE', 'UNSLOTH_ENABLE_LOGGING',
            'UNSLOTH_RETURN_LOGITS', 'TORCH_COMPILE_DISABLE')})
    return hardware, kernels


class DeviceComputeTimer:
    """Use CUDA events only in an explicitly requested bounded profiling run."""

    def __init__(self, torch):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()

    def seconds(self):
        # The loop's existing safety callback already synchronizes the device.
        if not self.end_event.query():
            raise ValueError('Device timing requires the completed safety synchronization')
        return self.start_event.elapsed_time(self.end_event) / 1000
