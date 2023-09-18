# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import timesformer.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)

from .ssim_eval import ssim_select, load_ssim_file, ssim_select_aux, select_and_sort_videos
from .aligned_sampler import calc_batch_jumps

@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        self._is_video_caching = utils.is_video_caching(cfg, mode)
        self._is_ssim_caching = utils.is_ssim_caching(cfg, mode)
        self._is_caching = self._is_video_caching or self._is_ssim_caching
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"] or self._is_caching:
            # FIXME: Detect if it is in caching by the file existence.
            #        Unless you want to cache every crops.
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = {}
        videos_info_arr = []
        with PathManager.open(path_to_file, "r") as f:
            for path_label in f.read().splitlines():
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                _, video_name = self.get_video_dir_name(path)
                self._path_to_videos[video_name] = path
                videos_info_arr.append([video_name, label])

        self.ssim_loaded_flag = False
        path_to_ssim_file = utils.get_ssim_filename(self.cfg, self.mode)
        if self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.ENABLED and self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.PRE_SSIM and PathManager.exists(path_to_ssim_file):
            self.ssim_loaded_flag = True
            logger.info("{} split SSIM Data Loading ...".format(self.mode))
            vid_ssim = load_ssim_file(path_to_ssim_file)
            logger.info("{} split SSIM Sorting ...".format(self.mode))
            videos_info_arr = select_and_sort_videos(videos_info_arr, vid_ssim, self.cfg)
            logger.info("{} split SSIM data loaded.".format(self.mode))
            self.batch_steps_idx = calc_batch_jumps(videos_info_arr, self._num_clips, reverse=self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.REVERSE)
            logger.info("{} split batch size is available. Batch width jumps at {}.".
                        format(self.mode, self.batch_steps_idx))

        new_paths = []
        self._video_names = []
        self._rel_path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._ssim_values = []
        self._ssim_selections = []
        for clip_idx, path_label in enumerate(videos_info_arr):
            video_name, label = path_label[0], path_label[1]
            for idx in range(self._num_clips):
                self._video_names.append(video_name)
                path = self._path_to_videos[video_name]
                new_paths.append(
                    os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                )
                self._rel_path_to_videos.append(path)
                self._labels.append(int(label))
                self._spatial_temporal_idx.append(idx)
                self._video_meta[clip_idx * self._num_clips + idx] = {}
                if self.ssim_loaded_flag:
                    self._ssim_values.append(path_label[2])
                    self._ssim_selections.append(path_label[3])
        self._path_to_videos = new_paths

        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def get_batch_steps_idx(self):
        """
        Get the batch jump indices indicating the width will increase since this border.
        Returns:
            indices (array|None): If it is available return the indices array.
                Otherwise, return None indicating an error.
        """
        if self.ssim_loaded_flag:
            return self.batch_steps_idx
        else:
            return None

    def get_video_dir_name(self, video_path: str):
        video_dir, video_name = video_path.rsplit('/', 1)
        return video_dir, video_name[:11]

    def get_cached_path(self, video_path: str, format="npz"):
        """
        Get the cached paths for the video_path.
        Args:
            video_path (str): the original video_path.
            frame_num (int): the number of frames.
            sampling_rate (int): the sampling rate.
        Returns:
            paths (list): Required picture lists.
        """
        video_dir, video_name = self.get_video_dir_name(video_path)
        video_pic_dir = os.path.join(
            utils.get_video_cache_dir(self.cfg),
            video_dir
        )
        if not PathManager.exists(video_pic_dir):
            PathManager.mkdirs(video_pic_dir)
        if format == "npz" or format == "npz_compressed":
            return "{}.npz".format(os.path.join(video_pic_dir, video_name))
        elif format == "jpg":
            return ["{}_{:06d}.jpg".format(os.path.join(video_pic_dir, video_name), i)
                    for i in range(self.cfg.DATA.NUM_FRAMES)]
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):

            if self.cfg.DATA.CACHE:
                # Cached path for the video
                cached_path = self.get_cached_path(self._rel_path_to_videos[index], self.cfg.DATA.CACHE_FORMAT)

            if self.cfg.DATA.CACHE and PathManager.exists(cached_path[0] if isinstance(cached_path, list) else cached_path):
                # Read the cache. Whether it is cached is based on the first frame for simplicity.
                frames = utils.load_frames(cached_path, format=self.cfg.DATA.CACHE_FORMAT)
            else:
                # If the pictures do not exist:
                # decode the video and save them for the next round.
                video_container = None
                try:
                    video_container = container.get_video_container(
                        self._path_to_videos[index],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    logger.warning(
                        "Failed to meta load video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                # Decode video. Meta info is used to perform selective decoding.
                frames = decoder.decode(
                    video_container,
                    sampling_rate,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,
                )

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames is None:
                    logger.warning(
                        "Failed to decode video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                if self.cfg.DATA.CACHE:
                    # Save those images for the next round. If cache is enabled.
                    utils.save_frames(cached_path, frames, format=self.cfg.DATA.CACHE_FORMAT)

            label = self._labels[index]
            meta = {}

            if self._is_caching:
                # Normal video caching doesn't require any meta.
                # FIXME: video caching will also get proceed with batch size=1 to reduce further unnecessary crops.
                if self._is_ssim_caching:
                    # Won't calculate the SSIM values, they are handled in the preprocessing.
                    # Batch size is 1 and no normalization and crops.
                    # Video Name is only used in ssim caching.
                    meta['video_name'] = self._video_names[index]
            else:
                # Do the normal job for enabling stacking batches.
                if self.mode in ["test"] and self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.ENABLED and not self.ssim_loaded_flag:
                    # TODO: Currently only in test mode.
                    # Evaluate SSIM values later.
                    # Pass the frames to frontend to process.
                    meta['frames'] = frames

                # Perform color normalization.
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )

                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

                if self.mode in ["test"] and self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.ENABLED and self.ssim_loaded_flag:
                    # TODO: Currently only in test mode.
                    # We make index selection here to avoid index selecting on HD videos.
                    # Load the preloaded SSIM data.
                    ssim_meta = self._ssim_values[index]
                    ssim_selection = self._ssim_selections[index]
                    meta['ssim_selected'] = len(ssim_selection)
                    if self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK:
                        ssim_selection = ssim_selection + ssim_select_aux(ssim_meta, ssim_selection,
                                                                          self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK_TRIM_TAIL,
                                                                          self.cfg.DATA.SSIM_ADAPTIVE_SAMPLING.FALLBACK_AUX_ALL)
                    frames = torch.index_select(frames, dim=1, index=torch.Tensor(ssim_selection).long())

                if not self.cfg.MODEL.ARCH in ['vit']:
                    frames = utils.pack_pathway_output(self.cfg, frames)
                else:
                    # Perform temporal sampling from the fast pathway.
                    # frames = torch.index_select(
                    #      frames,
                    #      1,
                    #      torch.linspace(
                    #          0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES
                    #
                    #      ).long(),
                    # )
                    pass

            return frames, label, index, meta
        else:
            video_path = self._path_to_videos[index]
            msg = "FAILED to fetch video idx {} from {} after {} retries.".format(
                        index, video_path, self._num_retries
                    )
            if self._is_caching:
                logger.error(msg)       # won't raise error if it is caching.
                return -1, -1, -1, {'failed_path': video_path}
            else:
                raise RuntimeError(msg)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
