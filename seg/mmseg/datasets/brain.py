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


@DATASETS.register_module()
class BrainDataset(CustomDataset):
    CLASSES = ('1', '2', '3', '4', '5', '6',
               '7', '8', '9', '10', '11',
               '12', '13', '14', '15')

    PALETTE = [[153, 153, 153], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70]]
    
    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(BrainDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)

        self.foreground_idx_start = 1

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
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
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        # WandbLogPredictions(results, gt_seg_maps)

        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)

        # WandbLogImages(results, gt_seg_maps, num_classes)
        # quit()
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        ret_metrics_summary_foreground = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value[self.foreground_idx_start:]) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items() if ret_metric != 'aAcc'
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        summary_table_data_foreground = PrettyTable()
        for key, val in ret_metrics_summary_foreground.items():
            if key == 'aAcc':
                summary_table_data_foreground.add_column(key, [val])
            else:
                summary_table_data_foreground.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)
        print_log('Summary foreground:', logger)
        print_log('\n' + summary_table_data_foreground.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0
        
        for key, value in ret_metrics_summary_foreground.items():
            if key == 'aAcc':
                eval_results[key + '-foreground'] = value / 100.0
            else:
                eval_results['m' + key + '-foreground'] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results



@DATASETS.register_module()
class WMHDataset(CustomDataset):
    CLASSES = ('1', '2')

    PALETTE = [[153, 153, 153], [128, 64, 128]]

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(WMHDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
        
        self.foreground_idx_start = 1

@DATASETS.register_module()
class WMHDatasetBCG(CustomDataset):
    CLASSES = ('1', '2', '3')

    PALETTE = [[153, 153, 153], [128, 64, 128], [244, 35, 232]]

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(WMHDatasetBCG, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
        
        self.foreground_idx_start = 2