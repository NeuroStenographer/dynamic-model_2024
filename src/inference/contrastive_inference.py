import torch
import random
import math
from itertools import product
from torch.nn import functional as F

def temporal_ica_inference(model, loss_fn, batch, n_neg_keys=20, gap=30):
    # if not loss_fn.negative_mode != 'paired':
    #     raise ValueError('negative_mode must be "paired"')
    # sample n_neg_keys from plus or minus gap away from the current index
    # ensure all negative keys are from an index less than i-gap or greater than i+gap
    # take n_neg_keys random indexes from the range [0, i-gap) and (i+gap, len(projections))
    signal = batch['signals']
    if isinstance(signal, torch.Tensor):
        proj = model({'input': signal})
    else:
        raise ValueError(f'signal must be a torch.Tensor, got {type(signal)}')
    projection = proj["FrameProjector"].squeeze(0)
    queries = projection[:-1,:]
    pos_keys = projection[1:,:]
    neg_keys = compute_neg_keys(projection, n_neg_keys, gap, mode="offset")

    return loss_fn(queries, pos_keys, neg_keys)



def simclr_inference(model, loss_fn, batch, n_neg_keys=20, gap=30):
    """Inference function for augmented SimCLR models.
    """
    signals = batch['signals']

    # augment the signals twice
    aug_signals_1 = augment_signals(signals)
    aug_signals_2 = augment_signals(signals)

    queries = model({'input': aug_signals_1})['FrameProjector'].squeeze(0)
    pos_keys = model({'input': aug_signals_2})['FrameProjector'].squeeze(0)
    neg_keys = compute_neg_keys(pos_keys, n_neg_keys, gap, mode="no_offset")

    loss = loss_fn(queries, pos_keys, neg_keys)

    return loss


def augment_signals(x, max_shift=2, max_rotate=10):
    """Apply augmentations to each frame in the batch.

    Args:
        x (torch.Tensor): Batch tensor of shape (N, T, A, C, H, W).
        max_shift (int): Maximum pixel shift for shift-rotate augmentation.
        max_rotate (int): Maximum rotation for shift-rotate augmentation.

    Returns:
        torch.Tensor: The augmented batch.
    """
    N, T, A, C, H, W = x.shape
    new_x = x.clone()

    # Iterate over each frame in the batch
    for n, t, a in product(range(N), range(T), range(A)):
                    frame = x[n, t, a, :, :, :]
                    # Apply shift-rotate augmentation
                    new_frame = shift_rotate_frame(frame, max_shift, max_rotate)
                    x[n, t, a, :, :, :] = new_frame
    return new_x

def shift_rotate_frame(frame, max_shift=2, max_rotate=3, padding_mode='reflection'):
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

def compute_neg_keys(projection, n_neg_keys, gap, mode='offset'):
    neg_keys = []
    T = projection.shape[0]
    if mode=='offset':
        rng = T-1
    elif mode=='no_offset':
        rng = T
    for t in range(rng):
        neg_indexes = []
        for _ in range(n_neg_keys):
            # get a random index on the left or right of i
            # case 1. There are less than gap indexes on the left of i
            # case 2. There are less than gap indexes on the right of i
            # case 3. There are more than gap indexes on both sides of i
            if t < gap:
                neg_index = random.randint(gap, T-1)
            elif t > T - gap:
                neg_index = random.randint(0, T-gap)
            else:
                p_left = (t-gap) / T
                p_right = (T-t-gap) / T
                # pick a random side
                side = random.choices(['left', 'right'], weights=[p_left, p_right])[0]
                if side == 'left':
                    neg_index = random.randint(0, t-gap)
                else:
                    neg_index = random.randint(t+gap, T-1)
            neg_indexes.append(neg_index)
        # reshape to (T, M, F) where M is the number of negative keys
        neg_keys_t = projection[neg_indexes,:]
        neg_keys_t = neg_keys_t
        neg_keys.append(neg_keys_t)
    neg_keys = torch.stack(neg_keys, dim=0)
    return neg_keys