# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .brain import BrainDataset


@DATASETS.register_module()
class prostateDataset(BrainDataset):
    CLASSES = ('1', '2', '3')

    PALETTE = [[0, 0, 0], [153, 153, 153], [128, 64, 128], [244, 35, 232]]

    VOLUME_SIZE = 48
    rescale_masks = True
    resolution_proc =  (3, 1, 1)
    metric_version = 'new'

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(BrainDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            ignore_index=0,
            **kwargs)
        
        self.foreground_idx_start = 2