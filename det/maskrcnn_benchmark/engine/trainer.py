# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def do_da_train(
    model,
    source_data_loader,
    target_data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    cfg
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()
        images = (source_images+target_images).to(device)
        targets = [target.to(device) for target in list(source_targets+target_targets)]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def do_mask_da_train(
    model, model_teacher,
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
    checkpointer_teacher
):
    from maskrcnn_benchmark.structures.image_list import ImageList
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    logger.info("with_MIC: On")
    meters = MetricLogger(delimiter=" ")
    max_iter = len(source_data_loader)
    start_iter = arguments["iteration"]
    model.train()
    model_teacher.eval()
    start_training_time = time.time()
    end = time.time()
    for iteration, ((source_images, source_targets, idx1), (target_images, target_targets, idx2)) in enumerate(zip(source_data_loader, target_data_loader), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        source_images = source_images.to(device)
        target_images = target_images.to(device)
        images = source_images + target_images
        targets = [target.to(device) for target in list(source_targets + target_targets)]

        # generate pseudo labels for masked target image
        masked_target_images = masking(target_images.tensors.clone().detach()).detach()
        model_teacher.update_weights(model, iteration)
        target_output = model_teacher(target_images)
        target_pseudo_labels, pseudo_masks = process_pred2label(target_output, threshold=cfg.MODEL.PSEUDO_LABEL_THRESHOLD)

        record_dict = model(images, targets)

        # apply pseudo label on masked images
        if len(target_pseudo_labels)>0:
            masked_images = ImageList(masked_target_images[pseudo_masks], target_images.image_sizes)
            masked_taget = target_pseudo_labels
            masked_loss_dict = model(masked_images, masked_taget, use_pseudo_labeling_weight=cfg.MODEL.PSEUDO_LABEL_WEIGHT, with_DA_ON=False)
            
            new_record_all_unlabel_data = {}
            for key in masked_loss_dict.keys():
                new_record_all_unlabel_data[key + "_mask"] = masked_loss_dict[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

        # weight losses
        loss_dict = {}
        for key in record_dict.keys():
            if key.startswith("loss"):
                if key == "loss_box_reg_mask" or key == "loss_rpn_box_reg_mask":
                    # pseudo bbox regression <- 0
                    loss_dict[key] = record_dict[key] * 0
                elif key.endswith('_mask') and 'da' in key:
                    loss_dict[key] = record_dict[key] * 0
                elif key == 'loss_classifier_mask' or key == 'loss_objectness_mask':
                    loss_dict[key] = record_dict[key] * cfg.MODEL.PSEUDO_LABEL_LAMBDA
                else:  # supervised loss
                    loss_dict[key] = record_dict[key] * 1

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration != 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            # checkpointer_teacher.save("model_teacher_{:07d}".format(iteration), **arguments)
        if iteration == max_iter-1:
            checkpointer.save("model_final", **arguments)
            checkpointer_teacher.save("model_final_teacher", **arguments)
        if torch.isnan(losses_reduced).any():
            logger.critical('Loss is NaN, exiting...')
            return 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def process_pred2label(target_output, threshold=0.7):
    from maskrcnn_benchmark.structures.bounding_box import BoxList
    pseudo_labels_list = []
    masks = []
    for idx, bbox_l in enumerate(target_output):
        pred_bboxes = bbox_l.bbox.detach()
        labels = bbox_l.get_field('labels').detach()
        scores = bbox_l.get_field('scores').detach()
        # print(torch.max(scores))
        filtered_idx = scores>=threshold
        filtered_bboxes = pred_bboxes[filtered_idx]
        filtered_labels = labels[filtered_idx]
        new_bbox_list = BoxList(filtered_bboxes, bbox_l.size, mode=bbox_l.mode)
        new_bbox_list.add_field("labels", filtered_labels)
        domain_labels = torch.ones_like(filtered_labels, dtype=torch.uint8).to(filtered_labels.device)
        new_bbox_list.add_field("is_source", domain_labels)

        if len(new_bbox_list)>0:
            pseudo_labels_list.append(new_bbox_list)
            masks.append(idx)
    return pseudo_labels_list, masks