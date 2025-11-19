import torch
import torch.nn.functional as F
import math



def causal_temporal_smoothing(tensor, window_size, sigma=1.0):
    """
    Apply causal temporal smoothing along the time dimension of a tensor.

    Args:
    tensor (torch.Tensor): A tensor of shape (N, T, A, C, H, W).
    window_size (int): The size of the window for smoothing.
    sigma (float): Standard deviation for Gaussian kernel.

    Returns:
    torch.Tensor: Tensor after applying causal temporal smoothing along the time dimension.
    """
    # Create a Gaussian kernel for smoothing
    kernel = torch.arange(window_size, dtype=torch.float32) - (window_size - 1)
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Reshape kernel to match the dimensions: (out_channels, in_channels, kernel_size)
    kernel = kernel.view(1, 1, -1).repeat(tensor.shape[2], 1, 1)

    # Apply padding to the left (causal) and no padding to the right
    padding = (window_size - 1, 0)

    # Apply 1D convolution along the time dimension
    # tensor shape: (N, T, A, C, H, W) -> (N*A*C*H*W, T) for convolution
    N, T, A, C, H, W = tensor.shape
    tensor_reshaped = tensor.view(N * A * C * H * W, T)
    smoothed_tensor = F.conv1d(tensor_reshaped.unsqueeze(1), kernel, padding=padding)

    # Reshape tensor back to original shape
    return smoothed_tensor.view(N, T, A, C, H, W)

def rolling_normalize_batch(batch, window_size):
    """Uses weighted sum of previous mean and sd to normalize current batch.
    After the window size is reached, the previous mean and sd are no longer used.
    """
    N, T, A, C, H, W = batch.shape
    normalized_batch = torch.zeros_like(batch)
    for n in range(N):
        for a in range(A):
            for c in range(C):
                        cur_array = batch[n, :, a, c, :, :]
                        prev_mean = torch.mean(cur_array[:window_size])
                        prev_sd = torch.std(cur_array[:window_size])
                        normalized_array = rolling_normalize_array(cur_array, prev_mean, prev_sd, window_size)
                        normalized_batch[n, :, a, c, :, :] = normalized_array

def rolling_normalize_array(cur_array, prev_mean, prev_sd, window_size):
    """Uses weighted sum of previous mean and sd to normalize current array.
    After the window size is reached, the previous mean and sd are no longer used.
    """
    T = cur_array.shape[0]
    normalized_array = torch.zeros_like(cur_array)
    for t in range(T):
        if t < window_size:
            cur_mean = torch.mean(cur_array[:t])
            cur_sd = torch.std(cur_array[:t])
            proportion = (t + 1) / window_size
            mean = proportion * cur_mean + (1 - proportion) * prev_mean
            sd = proportion * cur_sd + (1 - proportion) * prev_sd
        else:
            mean = torch.mean(cur_array[t-window_size:t])
            sd = torch.std(cur_array[t-window_size:t])
        normalized_array[t] = (cur_array[t] - mean) / (sd + 1e-8)


# implement unit testing
def time_warp_array(tensor, warp_factor_range):
    """
    Apply time warping to a one-sentence tensor of shape (1, T, A, C, H, W).

    Args:
    - tensor (torch.Tensor): A tensor of shape (1, T, A, C, H, W).
    - warp_factor_range (float, float): A tuple indicating the min and max range for the warp factor.

    Returns:
    - torch.Tensor: A time-warped tensor.
    """
    _, time_steps, *other_dims = tensor.size()
    warp_factor = torch.empty(1).uniform_(*warp_factor_range).to(tensor.device)

    # Generate warped time indices
    warped_time_steps = (torch.arange(time_steps).float() * warp_factor).long()
    warped_time_steps = torch.clamp(warped_time_steps, 0, time_steps - 1)

    # Apply time warping
    warped_tensor = torch.gather(tensor, 1, warped_time_steps.view(1, -1, 1, 1, 1, 1).expand(-1, -1, *other_dims))

    return warped_tensor


def augment_batch(x, max_shift=2, max_rotate=10):
    """Apply augmentations to each frame in the batch.

    Args:
        x (torch.Tensor): Batch tensor of shape (N, T, A, C, H, W).
        max_shift (int): Maximum pixel shift for shift-rotate augmentation.
        max_rotate (int): Maximum rotation for shift-rotate augmentation.

    Returns:
        torch.Tensor: The augmented batch.
    """
    N, T, A, C, H, W = x.shape()
    new_x = x.clone()

    # Iterate over each frame in the batch
    for n in range(N):
        for a in range(A):
            for t in range(T):
                    frame = x[n, t, a, :, :, :]
                    # Apply shift-rotate augmentation
                    new_frame = shift_rotate_frame(frame, max_shift, max_rotate)
                    new_frame = add_spike_pow_noise_and_bias(new_frame)
                    new_x[n, t, a] = new_frame

    return new_x

def shift_rotate_frame(frame, max_shift=2, max_rotate=10, padding_mode='reflection'):
    """Apply a small random shift and rotation to a single frame.

    Args:
        frame (torch.Tensor): A single frame tensor of shape (C, H, W).
        max_shift (int): Maximum pixel shift.
        max_rotate (int): Maximum rotation in degrees.

    Returns:
        torch.Tensor: The transformed frame.
    """
    C, H, W = frame.size()

    # Sample random shift and rotation values
    shift_x = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    shift_y = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    rotate_angle = torch.randint(-max_rotate, max_rotate + 1, (1,)).item()

    # Convert rotation angle to radians and create affine transformation matrix
    theta = rotate_angle * math.pi / 180
    transformation_matrix = torch.tensor([
        [math.cos(theta), -math.sin(theta), shift_x],
        [math.sin(theta), math.cos(theta), shift_y]
    ], dtype=torch.float)

    # Apply grid sample
    grid = F.affine_grid(transformation_matrix.unsqueeze(0), frame.unsqueeze(0).size(), align_corners=True)
    new_frame = F.grid_sample(frame.unsqueeze(0), grid, padding_mode=padding_mode, align_corners=True)

    return new_frame.squeeze(0)

def add_spike_pow_noise_and_biase(frame, max_bias=1, max_std=1):
    """Adds salt and pepper noise to a single frame.
    Only applies to the first channel of the frame. The other channels are
    left untouched.
    """
    C, H, W = frame.size()

    # Sample random bias and standard deviation values
    bias = torch.randint(-max_bias, max_bias + 1, (1,)).item()
    std = torch.randint(0, max_std + 1, (1,)).item()
    # Add bias and noise
    new_frame = frame.clone()
    new_frame[0] += bias
    new_frame[0] += torch.randn((H, W)) * std
    return new_frame
