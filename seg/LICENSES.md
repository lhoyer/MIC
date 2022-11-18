MIC: Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.

MIC for semantic segmentation is released under the Apache License 2.0 ([resources/license_mic](resources/license_mic)),
while some specific features in this repository are with other licenses.
Users should be careful about adopting these features in any commercial matters.

- SegFormer and MixTransformer: Copyright (c) 2021, NVIDIA Corporation,
  licensed under the NVIDIA Source Code License ([resources/license_segformer](resources/license_segformer))
    - [mmseg/models/decode_heads/segformer_head.py](mmseg/models/decode_heads/segformer_head.py)
    - [mmseg/models/backbones/mix_transformer.py](mmseg/models/backbones/mix_transformer.py)
    - configs/\_base\_/models/segformer*
- AdaptSegNet: licensed for non-commercial research purposes only ([resources/license_adaptsegnet](resources/license_adaptsegnet))
    - [mmseg/models/uda/fcdiscriminator.py](mmseg/models/uda/fcdiscriminator.py)
    - [mmseg/models/uda/advseg.py](mmseg/models/uda/advseg.py)
- DACS: Copyright (c) 2020, vikolss,
  licensed under the MIT License ([resources/license_dacs](resources/license_dacs))
    - [mmseg/models/utils/dacs_transforms.py](mmseg/models/utils/dacs_transforms.py)
    - parts of [mmseg/models/uda/dacs.py](mmseg/models/uda/dacs.py)
- Advent: Copyright (c) 2019, Valeo,
  licensed under the Apache License, Version 2.0 ([resources/license_advent](resources/license_advent))
    - parts of [mmseg/models/uda/minent.py](mmseg/models/uda/minent.py)

This repository is based on:
- HRDA: Copyright (c) 2022, ETH Zurich, Lukas Hoyer,
  licensed under the Apache License, Version 2.0 ([resources/license_hrda](resources/license_hrda))
- DAFormer: Copyright (c) 2021-2022, ETH Zurich, Lukas Hoyer,
  licensed under the Apache License, Version 2.0 ([resources/license_daformer](resources/license_daformer))
- MMSegmentation v0.16: Copyright (c) 2020, The MMSegmentation Authors,
  licensed under the Apache License, Version 2.0 ([resources/license_mmseg](resources/license_mmseg))
