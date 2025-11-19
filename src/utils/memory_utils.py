import torch

def estimate_memory_usage(SHARED_COMPONENTS):
    total_params = 0

    for component in SHARED_COMPONENTS.values():
        # Count the number of parameters in each component
        num_params = sum(p.numel() for p in component.parameters() if p.requires_grad)
        total_params += num_params

    # Assuming float32 for each parameter, each parameter requires 4 bytes
    memory_bytes = total_params * 4

    # Convert bytes to megabytes (1 MB = 1024 squared bytes)
    memory_megabytes = memory_bytes / (1024 ** 2)

    return memory_megabytes


def get_current_memory_usage():
    # Get the current GPU usage
    return torch.cuda.memory_allocated() / (1024 ** 2)

def gpu_cpu_switching(forward_func):
    """A wrapper function """