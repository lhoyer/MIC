# Obtained from: https://github.com/lhoyer/HRDA
# Modifications: Add MIC options
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# uda settings
uda = dict(
    type='MinEnt',
    lambda_ent=dict(main=0.001, aux=0.0002),
    debug_img_interval=1000,
    # MIC params (inactive as mask_mode=None)
    alpha=0.999,
    mask_alpha='same',
    mask_pseudo_threshold='same',
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode=None,
    mask_lambda=0,
    mask_generator=None,
)
