import os
from fvcore.common.file_io import PathManager
from timesformer.datasets.ssim_eval import ssim_evaluation_mod, ssim_statistics, write_ssim_file
from timesformer.datasets import utils
from timesformer.datasets import loader
from timesformer.utils.meters import PreMeter
from timesformer.models.build import build_ssim
from pytorch_msssim import SSIM

import timesformer.utils.logging as logging
logger = logging.get_logger(__name__)

def preprocess(cfg, mode, writer=None):
    is_video_caching = utils.is_video_caching(cfg, mode)
    is_ssim_caching = utils.is_ssim_caching(cfg, mode)
    is_caching = is_video_caching or is_ssim_caching

    if not is_caching:
        logger.info("Caching has been completed! The step is skipped.")
        if writer is not None:
            # For the sake of fairness, start the clock here.
            writer.add_scalars({
                'perf/iter_time': 0,
                'perf/data_time': 0,
                'perf/net_time': 0
            }, global_step=-1)
    else:
        # Perform video caching or SSIM caching, or both.
        # batch size is 1 for all caching process (it is pointless to enable batches on decoding).
        pre_loader = loader.construct_loader(cfg, mode)
        pre_meter = PreMeter(len(pre_loader))

        if is_video_caching:
            caching_dir = utils.get_video_cache_dir(cfg)
            if not PathManager.exists(caching_dir):
                PathManager.mkdirs(caching_dir)
            # Clear the directory if the last attempt failed.
            # (The reason is the uncompleted decoding, no END file)
            # If the video caching step is performed correctly, then continue.
            logger.info("Cleaning...")
            for root, dirs, files in os.walk(caching_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        # Store the failed videos.
        failed_paths = []

        if is_ssim_caching:
            # Do SSIM evaluation (maybe together with decoding).
            _ssim = SSIM(size_average=False)
            _ssim = build_ssim(_ssim, cfg)

            pre_meter.iter_tic()
            logger.info("SSIM calculation start!")
            ssim_results = []
            for cur_iter, (inputs, labels, video_idx, meta) in enumerate(pre_loader):
                assert len(inputs) == 1
                if 'failed_path' in meta.keys():
                    pre_meter.data_toc()
                    failed_paths.append(meta['failed_path'][0])
                    pre_meter.iter_toc()
                else:
                    if cfg.NUM_GPUS:
                        # Transfer the data to the current GPU device.
                        if isinstance(inputs, (list,)):
                            for i in range(len(inputs)):
                                inputs[i] = inputs[i].cuda(non_blocking=True)
                        else:
                            inputs = inputs.cuda(non_blocking=True)

                    pre_meter.data_toc()
                    ssim_val = ssim_evaluation_mod(inputs[0], _ssim, cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FROM_MID).tolist()
                    pre_meter.iter_toc()
                    ssim_results.append([meta['video_name'][0], ssim_val])
                pre_meter.log_iter_stats(cur_iter)
                if writer is not None:
                    writer.add_scalars({
                        'perf/iter_time': pre_meter.iter_timer_aux.seconds(),
                        'perf/data_time': pre_meter.data_timer_aux.seconds(),
                        'perf/net_time': pre_meter.net_timer_aux.seconds()
                    }, global_step=cur_iter - pre_meter.overall_iters)
                pre_meter.iter_tic()

            # Cache in the mem till everything is all right.
            write_ssim_file(utils.get_ssim_filename(cfg, mode), ssim_results)
            avg_result, recommended_alpha = ssim_statistics(ssim_results, m=cfg.DATA.SSIM_ADAPTIVE_SAMPLING.SELECT_MIN)
            logger.info("Histogram for ssim_avg (Removed 0): {}".format(avg_result))
            logger.info("Recommended Alpha: {}".format(recommended_alpha))
        else:
            # Only perform decoding.
            pre_meter.iter_tic()
            logger.info("Video caching start!")
            for cur_iter, (inputs, labels, video_idx, meta) in enumerate(pre_loader):
                assert len(inputs) == 1
                if 'failed_path' in meta.keys():
                    failed_paths.append(meta['failed_path'][0])
                pre_meter.data_toc()
                pre_meter.log_iter_stats(cur_iter)
                pre_meter.iter_toc()
                # No calculation.
                pre_meter.iter_tic()

        if is_video_caching:
            # Write the "END" sign if it is video caching.
            if len(failed_paths) > 0:
                raise RuntimeError("Remove the following video path in the list before continuing:\n  {}".format(
                    '\n  '.join(failed_paths)
                ))
            with PathManager.open(utils.get_cache_end_file(cfg, mode), "w") as f:
                pass
        pre_meter.finalize_metrics()
