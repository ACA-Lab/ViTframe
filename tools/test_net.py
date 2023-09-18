# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import cv2
from einops import rearrange, reduce, repeat
import scipy.io
from .preprocessing import preprocess

import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.misc as misc
import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TestMeter

from timesformer.models.build import build_ssim
from pytorch_msssim import SSIM
from timesformer.datasets.ssim_eval import ssim_evaluation_mod, ssim_select, ssim_select_aux

logger = logging.get_logger(__name__)

from timesformer.datasets.ssim_eval import ssim_select_aux

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    if cfg.DATA.SSIM_ADAPTIVE_SAMPLING.ENABLED:
        # TODO: The overhead here is not decisive. It is not calculated.
        _ssim = SSIM(size_average=False)
        _ssim = build_ssim(_ssim, cfg)

    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):

        # inputs is N C T H W

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()

        aux_avail = False
        if cfg.DATA.SSIM_ADAPTIVE_SAMPLING.ENABLED:
            if 'ssim_selected' in meta.keys():
                prep_frame_num = inputs.shape[2]
                selected_num = meta['ssim_selected'][0]
                aux_avail = True if selected_num < prep_frame_num else False
                # Re-assemble the data on GPU
                if aux_avail:
                    inputs, aux_inputs = torch.split(inputs, [selected_num, prep_frame_num - selected_num], dim=2)
                test_meter.data_toc()
            else:
                # Evaluate SSIM on the fly.
                # FIXME: The batch_size should be 1.
                assert len(inputs) == 1
                if cfg.NUM_GPUS:
                    frames = meta['frames'].cuda()
                else:
                    frames = meta['frames']
                device = frames.device
                test_meter.data_toc()
                ssim_result = ssim_evaluation_mod(frames[0], _ssim, cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FROM_MID)
                ssim_selection = torch.Tensor(ssim_select(ssim_result, cfg.DATA.SSIM_ADAPTIVE_SAMPLING.SSIM_THRESHOLD, cfg.DATA.SSIM_ADAPTIVE_SAMPLING.SELECT_MIN)).long().to(device)
                aux_selection = torch.Tensor(ssim_select_aux(ssim_result, ssim_selection, cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK_TRIM_TAIL, cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK_AUX_ALL)).long().to(device)
                aux_avail = True if len(aux_selection) > 0 else False
                if aux_avail:
                    aux_inputs = torch.index_select(inputs, dim=2, index=aux_selection)
                inputs = torch.index_select(inputs, dim=2, index=ssim_selection)
        else:
            test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS:
                # meta is required to move to GPU only when dectection.
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            # Update iter stats and the frame number count.
            # inputs may be N C F H W
            test_meter.log_iter_stats(cur_iter, frame_add=[inputs.shape[2]]*inputs.shape[0])
        else:
            # Perform the forward pass.
            #inputs = inputs.cpu()
            preds = model(inputs)

            if cfg.DATA.SSIM_ADAPTIVE_SAMPLING and cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK:
                # In order to make the prediction additive
                # normalize the predictions.
                preds = torch.softmax(preds, dim=1)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_idx = du.all_gather(
                    [preds, labels, video_idx]
                )

            # Fallback if necessary. Most Calculation is on GPU.
            aux_frames_added = None
            if cfg.DATA.SSIM_ADAPTIVE_SAMPLING and aux_avail and cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK:
                aux_frames_added = [0] * inputs.shape[0]
                top2_values, top2_indices = torch.topk(preds, 2, dim=1, largest=True, sorted=True)
                fallback_ind = []
                for ind in range(preds.shape[0]):
                    if top2_values[ind][0] - top2_values[ind][1] < cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK_EPSILON:
                        fallback_ind.append(ind)
                if len(fallback_ind) > 0:
                    fallback_inputs = aux_inputs[fallback_ind]
                    preds_aux = model(fallback_inputs)
                    preds_aux = torch.softmax(preds_aux, dim=1)
                    if cfg.NUM_GPUS > 1:
                        # Gather the result from multiple machines.
                        preds_aux = du.all_gather(preds_aux)
                    for aux_ind, ind in enumerate(fallback_ind):
                        preds[ind] = (preds[ind] + preds_aux[aux_ind]) / 2
                        aux_frames_added[ind] = fallback_inputs[aux_ind].shape[1]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()
            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
            # Update iter stats and the frame number count.
            # inputs may be N C T H W
            test_meter.log_iter_stats(cur_iter,
                                      frame_add=[inputs.shape[2]]*inputs.shape[0],
                                      aux_frame_add=aux_frames_added
                                      )
            if writer is not None:
                writer.add_scalars({
                    'perf/iter_time': test_meter.iter_timer_aux.seconds(),
                    'perf/data_time': test_meter.data_timer_aux.seconds(),
                    'perf/net_time': test_meter.net_timer_aux.seconds(),
                    'perf/cpumem': misc.cpu_mem_usage()[0],
                    'perf/gpumem': misc.gpu_mem_usage(),
                }, global_step=cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with PathManager.open(save_path, "wb") as f:
                pickle.dump([all_labels, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    if cfg.DATA.CACHE and cfg.DATA.CACHE_ONLY:
        logger.info("Cache only, model build skipped.")
    else:
        # Build the video model and print model statistics.
        model = build_model(cfg)
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)

        cu.load_test_checkpoint(cfg, model)

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Preprocess if necessary (cache and SSIM)
    preprocess(cfg, "test", writer)
    if cfg.DATA.CACHE and cfg.DATA.CACHE_ONLY:
        logger.info("Cache only complete.")
        return

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        cfg.DATA.MULTI_LABEL,
        cfg.DATA.ENSEMBLE_METHOD,
    )

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
