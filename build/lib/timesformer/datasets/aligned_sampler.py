from torch.utils.data.sampler import Sampler, BatchSampler
import math


def calc_batch_jumps(videos_info_arr, scale_factor, reverse=True):
    batch_steps_idx = []
    current_batch_width = math.inf if reverse else 0
    for idx, info in enumerate(videos_info_arr):
        if (reverse and len(info[3]) < current_batch_width) or (not reverse and len(info[3]) > current_batch_width):
            current_batch_width = len(info[3])
            batch_steps_idx.append(idx * scale_factor)  # magnify
    return batch_steps_idx


class AlignedBatchSampler(BatchSampler):
    """
    Aligned Batch Sampler for data with meta.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool, batch_jumps) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.batch_jumps = batch_jumps

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            if idx in self.batch_jumps and len(batch) > 0:
                # A hit, current item will be in the next batch.
                if not self.drop_last or len(batch) == self.batch_size:
                    # drop_last tends to avoid incomplete batches.
                    yield batch
                batch = [idx]
            else:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
