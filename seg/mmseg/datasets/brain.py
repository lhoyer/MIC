# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .custom import CustomDataset


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