# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications: Add MIC options
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Baseline UDA
uda = dict(
    type='DACS',
    source_only=False,
    alpha=0.99,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode=None,
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=0,
    mask_generator=None,
    debug_img_interval=1000,
    print_grad_magnitude=False,
)
use_ddp_wrapper = True
