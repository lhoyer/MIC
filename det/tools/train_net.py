# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import random

import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train, do_da_train, do_mask_da_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.modeling.teacher import EMATeacher
from maskrcnn_benchmark.modeling.masking import Masking

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    # arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        source_data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_source=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        target_data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_source=False,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        if cfg.MODEL.MIC_ON:
            if cfg.MODEL.MASKING_AUGMENTATION:
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0.2, 0.2, True
            else:
                mask_color_jitter_p, mask_color_jitter_s, mask_blur = 0, 0, False
            model_t = build_detection_model(cfg)
            model_teacher = EMATeacher(model_t, alpha=cfg.MODEL.TEACHER_ALPHA).to(device)
            checkpointer_teacher = DetectronCheckpointer(
                cfg, model_teacher, None, None, output_dir, save_to_disk, last_checkpoint_name='last_checkpoint_teacher'
            )
            masking = Masking(
                block_size=cfg.MODEL.MASKING_BLOCK_SIZE,
                ratio=cfg.MODEL.MASKING_RATIO,
                color_jitter_s=mask_color_jitter_s,
                color_jitter_p=mask_color_jitter_p,
                blur=mask_blur,
                mean=cfg.INPUT.PIXEL_MEAN, 
                std=cfg.INPUT.PIXEL_STD)
            do_mask_da_train(
                model,
                model_teacher,
                source_data_loader,
                target_data_loader,
                masking,
                optimizer,
                scheduler,
                checkpointer,
                device,
                checkpoint_period,
                arguments,
                cfg,
                checkpointer_teacher,
            )
        else:
            do_da_train(
                model,
                source_data_loader,
                target_data_loader,
                optimizer,
                scheduler,
                checkpointer,
                device,
                checkpoint_period,
                arguments,
                cfg,
            )
    else:
        data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        
        do_train(
            model,
            data_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
        )

    return model


def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
    set_random_seed(args.seed, deterministic=args.deterministic)

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
