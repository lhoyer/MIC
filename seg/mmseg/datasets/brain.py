# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .custom import CustomDataset
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
from collections import OrderedDict
from functools import reduce
import mmcv
from mmseg.core import eval_metrics
import os

from skimage.transform import rescale
import wandb
from mmseg.models.utils.wandb_log_images import WandbLogPredictions
import pickle
from typing import Any, Dict

def convert_to_one_hot(mask, num_classes):
    """
    Convert a 3D segmentation mask to one-hot encoding.

    Parameters:
        mask (numpy.ndarray): The 3D segmentation mask of shape [D, H, W].
        num_classes (int): The number of unique class labels in the mask.

    Returns:
        numpy.ndarray: The one-hot encoded mask of shape [N, D, H, W].
    """
    # Ensure mask is of integer type
    mask = mask.astype(np.int32)

    # Initialize the one-hot encoded mask
    one_hot = np.zeros((num_classes, *mask.shape), dtype=np.int32)

    # Populate the one-hot encoded mask
    for i in range(num_classes):
        one_hot[i, :, :, :] = mask == i

    return one_hot


@DATASETS.register_module()
class BrainDataset(CustomDataset):
    CLASSES = (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    )

    PALETTE = [
        [153, 153, 153],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
    ]

    VOLUME_SIZE = 256
    metric_version = "new"

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(BrainDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix="_labelTrainIds.png",
            split=None,
            ignore_index=255,
            **kwargs,
        )

        self.foreground_idx_start = 1
        self.volume_meta = self.read_volume_meta()        

    def read_volume_meta(self):
        """Read image information from directory.

        Args:
            img_dir (str): Path to the image directory.
            data_prefix (str): Prefix of the data.

        Returns:
            list[dict]: Information of the image.
        """

        split = self.ann_dir.split("/")[-1]
        filename = f"{self.data_root}/scale_{split}.pickle"
        with open(filename, "rb") as f:
            return pickle.load(f)

    def get_scale_factor(self, patient_id: int) -> np.ndarray:
        """Calculate scale factor with respect to initial preprocessing.

        Args:
            patient_id (int): Index of the patient.
        Returns:
            np.ndarray: Scale factor of the image.
        """
        original_pix_size = np.array(
            [
                self.volume_meta["px"][patient_id],
                self.volume_meta["py"][patient_id],
                self.volume_meta["pz"][patient_id],
            ]
        )
        target_pix_size = self.volume_meta["resolution_proc"]
        scale_factor = np.array(original_pix_size) / np.array(target_pix_size)
        scale_factor[-1] = 1

        return scale_factor

    def evaluate(
        self,
        results,
        metric: str = "mIoU",
        logger=None,
        efficient_test: bool = False,
        with_online_evaluation: bool = False,
        rescale_masks: bool = False,
        **kwargs,
    ):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ["mIoU", "mDice", "mFscore"]
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError("metric {} is not supported".format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)

        if self.CLASSES is None:
            num_classes = len(reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        if self.metric_version == "old":
            print("Evaluating OLD metric!")
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
            )
        else:
            print("Evaluating NEW metric!")
            ret_metrics = dict()
            cur_vol = []
            cur_vol_gt = []
            cur_vol_id = 0
            for i in range(len(results)):
                result = results[i]  # .reshape(-1)
                gt_seg_map = gt_seg_maps[i]  # .reshape(-1)
                cur_vol.append(result)
                cur_vol_gt.append(gt_seg_map)
                if len(cur_vol) == self.VOLUME_SIZE:
                    mask_predicted = np.stack(cur_vol, axis=-1)
                    mask_true = np.stack(cur_vol_gt, axis=-1)

                    if rescale_masks:
                        scale_factor = self.get_scale_factor(cur_vol_id)
                        
                        mask_predicted = np.argmax(
                            rescale(
                                convert_to_one_hot(mask_predicted, num_classes),
                                scale_factor,
                                order=0,
                                preserve_range=True,
                                channel_axis=0,
                                mode="constant",
                            ).astype(np.uint8),
                            axis=0,
                        )

                        mask_true = np.argmax(
                            rescale(
                                convert_to_one_hot(mask_true, num_classes),
                                scale_factor,
                                order=0,
                                preserve_range=True,
                                channel_axis=0,
                                mode="constant",
                            ).astype(np.uint8),
                            axis=0,
                        )

                    ret_metrics_per_subject = eval_metrics(
                        mask_predicted,
                        mask_true,
                        num_classes,
                        self.ignore_index,
                        metric,
                        label_map=self.label_map,
                        reduce_zero_label=self.reduce_zero_label,
                    )

                    cur_vol = []
                    cur_vol_gt = []
                    cur_vol_id += 1

                    for k in ret_metrics_per_subject:
                        if k not in ret_metrics:
                            ret_metrics[k] = np.zeros(ret_metrics_per_subject[k].shape)
                        ret_metrics[k] += ret_metrics_per_subject[k]

                    if with_online_evaluation:
                        dice_per_class = dict(
                            zip(self.CLASSES, ret_metrics_per_subject["Dice"])
                        )
                        wandb.log({f"Dice per subject": dice_per_class})

            for k in ret_metrics:
                ret_metrics[k] = ret_metrics[k] / cur_vol_id

        if with_online_evaluation:
            dice_per_class = dict(zip(self.CLASSES, ret_metrics["Dice"]))
            wandb.log({f"Dice per subject total": dice_per_class})

        if with_online_evaluation:
            WandbLogPredictions(results, gt_seg_maps, self.PALETTE)

        # quit()
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict(
            {
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )

        ret_metrics_summary_foreground = OrderedDict(
            {
                ret_metric: np.round(
                    np.nanmean(ret_metric_value[self.foreground_idx_start :]) * 100, 2
                )
                for ret_metric, ret_metric_value in ret_metrics.items()
                if ret_metric != "aAcc"
            }
        )

        # each class table
        ret_metrics.pop("aAcc", None)
        ret_metrics_class = OrderedDict(
            {
                ret_metric: np.round(ret_metric_value * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            }
        )
        ret_metrics_class.update({"Class": class_names})
        ret_metrics_class.move_to_end("Class", last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == "aAcc":
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column("m" + key, [val])

        summary_table_data_foreground = PrettyTable()
        for key, val in ret_metrics_summary_foreground.items():
            if key == "aAcc":
                summary_table_data_foreground.add_column(key, [val])
            else:
                summary_table_data_foreground.add_column("m" + key, [val])

        print_log("per class results:", logger)
        print_log("\n" + class_table_data.get_string(), logger=logger)
        print_log("Summary:", logger)
        print_log("\n" + summary_table_data.get_string(), logger=logger)
        print_log("Summary foreground:", logger)
        print_log("\n" + summary_table_data_foreground.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == "aAcc":
                eval_results[key] = value / 100.0
            else:
                eval_results["m" + key] = value / 100.0

        for key, value in ret_metrics_summary_foreground.items():
            if key == "aAcc":
                eval_results[key + "-foreground"] = value / 100.0
            else:
                eval_results["m" + key + "-foreground"] = value / 100.0

        if with_online_evaluation:
            for k in eval_results:
                if "Dice" in k:
                    wandb.log({f"Final metrics/{k}": eval_results[k]})

        ret_metrics_class.pop("Class", None)
        for key, value in ret_metrics_class.items():
            eval_results.update(
                {
                    key + "." + str(name): value[idx] / 100.0
                    for idx, name in enumerate(class_names)
                }
            )

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results


@DATASETS.register_module()
class WMHDataset(BrainDataset):
    CLASSES = ("B", "Lesion")

    PALETTE = [[153, 153, 153], [128, 64, 128]]

    VOLUME_SIZE = 48
    metric_version = "new"

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(BrainDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix="_labelTrainIds.png",
            split=None,
            ignore_index=255,
            **kwargs,
        )

        self.foreground_idx_start = 1
        self.volume_meta = self.read_volume_meta()  


@DATASETS.register_module()
class WMHDatasetBCG(BrainDataset):
    CLASSES = ("B", "Brain", "Lesion")

    PALETTE = [[0, 0, 0], [153, 153, 153], [128, 64, 128]]

    VOLUME_SIZE = 48
    metric_version = "new"

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(BrainDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix="_labelTrainIds.png",
            split=None,
            ignore_index=255,
            **kwargs,
        )

        self.foreground_idx_start = 2
        self.volume_meta = self.read_volume_meta()  


@DATASETS.register_module()
class SpineMRIDataset(BrainDataset):
    CLASSES = ("B", "L1", "L2", "L3", "L4", "L5")

    PALETTE = [
        [153, 153, 153],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
    ]

    VOLUME_SIZE = 12
    metric_version = "new"

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(BrainDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix="_labelTrainIds.png",
            split=None,
            ignore_index=255,
            **kwargs,
        )

        self.foreground_idx_start = 1
        self.volume_meta = self.read_volume_meta()  


@DATASETS.register_module()
class SpineCTDataset(BrainDataset):
    CLASSES = ("B", "L1", "L2", "L3", "L4", "L5")

    PALETTE = [
        [153, 153, 153],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
    ]

    VOLUME_SIZE = 120
    metric_version = "new"

    def __init__(self, **kwargs):
        assert kwargs.get("split") in [None, "train"]
        if "split" in kwargs:
            kwargs.pop("split")
        super(BrainDataset, self).__init__(
            img_suffix=".png",
            seg_map_suffix="_labelTrainIds.png",
            split=None,
            ignore_index=255,
            **kwargs,
        )

        self.foreground_idx_start = 1
        self.volume_meta = self.read_volume_meta()
