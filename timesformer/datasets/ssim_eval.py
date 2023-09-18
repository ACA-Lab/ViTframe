import torch
import heapq
import numpy as np

# Import SSIM library for evaluation.
from pytorch_msssim import SSIM


def ssim_evaluation(frames: torch.Tensor):
    # Make it GPU device compatible
    device = frames.device
    _ssim = SSIM(size_average=False).to(device)
    return ssim_evaluation_mod(frames, _ssim)


def ssim_evaluation_mod(frames: torch.Tensor, _ssim: SSIM, from_mid=True):
    """
    Evaluate SSIM values between frames.
    Args:
        frames (Tensor) frames to be evaluated.
            The device of this variable will determine which device the computation will be on.
            The frames could be 5-dim or 4-dim.
        _ssim (SSIM): SSIM model.
    Return:
        ssim_result (Tensor): SSIM result.
    """
    # (N) T H W C -> (N) T C H W
    # Change dtype to float for SSIM window generation
    eval_frames = frames.transpose(-1, -3).float()

    # or frame_num
    frame_length = eval_frames.shape[-4]

    # Make it GPU device compatible
    device = frames.device

    # It is batch mode.
    is_batch = (len(eval_frames.shape) == 5)

    # Constant zero based on whether it is in batch mode
    const_zeros = torch.zeros(((eval_frames.shape[0],) if is_batch else ()) + (1,), device=device)

    if frame_length == 0:
        # Return empty.
        return torch.empty(((eval_frames.shape[0],) if is_batch else ()) + (0,), device=device)
    elif frame_length == 1:
        # The frame must be selected.
        ssim_result = const_zeros
    else:
        # Get SSIM results in the batch mode.
        # flatten those frames first then recover.
        eval_frames_pop_back = torch.narrow(eval_frames, dim=-4, start=0, length=frame_length-1)
        eval_frames_pop_front = torch.narrow(eval_frames, dim=-4, start=1, length=frame_length-1)
        if is_batch:
            eval_frames_pop_back = eval_frames_pop_back.flatten(start_dim=0, end_dim=1)
            eval_frames_pop_front = eval_frames_pop_front.flatten(start_dim=0, end_dim=1)
        ssim_result = _ssim(eval_frames_pop_back, eval_frames_pop_front)
        if is_batch:
            ssim_result = ssim_result.view(-1, frame_length-1)

        if from_mid:
            # The reference frame will be the middle of them,
            # and will be selected definitely.
            ref_idx = frame_length // 2

            # Insert the const_zeros to the ref_idx place.
            # len(ssim_result) == frame_length-1
            ssim_result_left, ssim_result_right = torch.split(ssim_result, [ref_idx, frame_length-1-ref_idx], dim=-1)
            ssim_result = torch.cat((ssim_result_left, const_zeros, ssim_result_right), dim=-1)
        else:
            # The first frame will always be selected.
            ssim_result = torch.cat((const_zeros, ssim_result), dim=-1)

    return ssim_result


def get_missing_index(rightend: int, selected: list):
    """find the missing indices among the two increasing arrays."""
    missing = []
    j = 0
    for i in selected:
        while j < i:
            missing.append(j)
            j += 1
        j += 1
    for k in range(j, rightend):
        missing.append(k)
    return missing


def get_smallest_idx(arr: list, idx_range: list, k: int):
    """
    Get the smallest idx from arr in idx_range. TopK fashion.
    """
    idx_val = [[j, arr[j]] for j in idx_range]
    # Sort all data, O(n * log(n))
    # idx_val = sorted(idx_val, key=lambda x: x[1])[:k]
    # In TopK fashion, O(n)
    idx_val = heapq.nsmallest(k, idx_val, lambda x: x[1])

    return [a[0] for a in idx_val]


def ssim_select(ssim_meta, alpha, select_min=1):
    selected_frame_idx = []
    for frame_idx, frame_ssim in enumerate(ssim_meta):
        if frame_ssim < alpha:
            selected_frame_idx.append(int(frame_idx))
    if len(selected_frame_idx) < select_min:
        # If it is less than the required number, select the minimum ones.
        # And reorder the indices of the two arrays
        # FIXME: This may be redundant with SSIM THRESHOLD.
        remaining_idx = get_missing_index(len(ssim_meta), selected_frame_idx)
        remaining_smallest_idx = get_smallest_idx(ssim_meta, remaining_idx, select_min - len(selected_frame_idx))
        selected_frame_idx = sorted(selected_frame_idx + remaining_smallest_idx)  # Keep the order.
    return selected_frame_idx


def ssim_select_aux(ssim_meta, selected_frame_idx, trim_tail=True, aux_all=False):
    frame_num = len(ssim_meta)
    selected_frame_num = len(selected_frame_idx)

    aux_frame_idx = get_missing_index(frame_num, selected_frame_idx)

    if aux_all:
        return aux_frame_idx

    if selected_frame_num * 2 > frame_num:
        # select no frame.
        return [] if trim_tail else aux_frame_idx

    # choose the frames with equal size
    # that has the smallest SSIM values
    # among the remaining frames.
    # NOTE: sort the indices in order!
    return sorted(get_smallest_idx(ssim_meta, aux_frame_idx, selected_frame_num))


def load_ssim_file(filepath):
    vid_ssim = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            cols = line.split()
            ssim_values = cols[1].split(',')
            ssim_values = [float(v) for v in ssim_values]
            vid_ssim[cols[0]] = ssim_values  # could contain _ for differentiate spatial_temporal_idx.
    return vid_ssim


def write_ssim_file(filepath, ssim_results):
    with open(filepath, "w") as f:
        for ssim_result in ssim_results:
            ssim_val = ','.join(["{:.6f}".format(v) for v in ssim_result[1]])
            f.write("{} {}\n".format(ssim_result[0], ssim_val))


def select_and_sort_videos(videos_info_arr, vid_ssim, cfg):
    for info in videos_info_arr:
        video_ssim = vid_ssim[info[0]]
        info.append(video_ssim)
        info.append(
            ssim_select(
                video_ssim,
                alpha=cfg.DATA.SSIM_ADAPTIVE_SAMPLING.SSIM_THRESHOLD,
                select_min=cfg.DATA.SSIM_ADAPTIVE_SAMPLING.SELECT_MIN,
            )
        )
    # NOTE: The sequence should be reversed in order to allocate all the memory.
    #       This could avoid the frequent memory allocation and cause memory exceeded.
    #       This is really important when the data is huge!
    #       Messy avoid this problem accidentally.
    videos_info_arr = sorted(videos_info_arr, key=lambda x: len(x[3]), reverse=cfg.DATA.SSIM_ADAPTIVE_SAMPLING.REVERSE)
    return videos_info_arr


def ssim_statistics(ssim_results, bins=20, m=1):
    frame_num = len(ssim_results[0][1])

    ssims = []
    for ssim_result in ssim_results:
        cur_ssim_result = ssim_result[1]
        cur_ssim_result.pop(frame_num // 2)
        ssims.extend(cur_ssim_result)

    ssim_distri = np.histogram(ssims, bins=bins, range=[0, 1])
    x_interval = 1 / bins
    prec = len(str(x_interval).split(".")[1])
    ssim_distri = [[round(k, prec), v] for k, v in zip(ssim_distri[1], ssim_distri[0])]

    p_alpha = (frame_num+m-2)/(2*frame_num-2)
    recommended_alpha = np.percentile(ssims, p_alpha*100)

    return ssim_distri, recommended_alpha
