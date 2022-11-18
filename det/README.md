# MIC for Domain-Adaptive Object Detection

## Getting started

### Installation

Please follow the instruction in [INSTALL.md](INSTALL.md) to install and use this repo.

For installation problems, please consult issues in [ maskrcnn-benchmark
](https://github.com/facebookresearch/maskrcnn-benchmark).

This code is tested under Debian 11 with Python 3.9 and PyTorch 1.12.0.

### Datasets

The datasets used in the repository can be downloaded from the following links:

* [Cityscapes and Foggy Cityscapes](https://www.cityscapes-dataset.com/)

The datasets should be organized in the following structure.
```
datasets/
├── cityscapes
│   ├── annotations
│   ├── gtFine
│   └── leftImg8bit
└── foggy_cityscapes
    ├── annotations
    ├── gtFine
    └── leftImg8bit_foggy
```

The annotations should be processed with [convert_cityscapes_to_coco.py](tools/cityscapes/convert_cityscapes_to_coco.py) and [convert_foggy_cityscapes_to_coco.py](tools/cityscapes/convert_foggy_cityscapes_to_coco.py) to be converted into coco format.

## Training

For experiments in our paper, we use the following script to run Cityscapes to Foggy Cityscapes adaptation task:

```shell
python tools/train_net.py --config-file configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml
```

## Testing

The trained model could be evaluated with the following script:
```shell
python tools/test_net.py --config-file "configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml" MODEL.WEIGHT <path_to_store_weight>/model_final.pth
```

## Checkpoints

Below, we provide the checkpoint of MIC(SADA) for Cityscapes→Foggy Cityscapes, which is used in the paper.

* [MIC(SADA) for Cityscapes→Foggy Cityscapes](https://drive.google.com/file/d/1AbsqakY1wtRzGYc6BBZ5W2eN0suLEDkR/view?usp=sharing)

## Where to find MIC in the code?

The most relevant files for MIC are:

* [configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml](configs/da_faster_rcnn/e2e_da_faster_rcnn_R_50_FPN_masking_cs.yaml):
  Definition of the experiment configurations in our paper.
* [tools/train_net.py](tools/train_net.py):
  Training script for UDA with MIC(sa-da-faster).
* [maskrcnn_benchmark/engine/trainer.py](maskrcnn_benchmark/engine/trainer.py):
  Training process for UDA with MIC(sa-da-faster).
* [maskrcnn_benchmark/modeling/masking.py](maskrcnn_benchmark/modeling/masking.py):
  Implementation of MIC.
* [maskrcnn_benchmark/modeling/teacher.py](maskrcnn_benchmark/modeling/teacher.py):
  Implementation of the EMA teacher.

## Acknowledgements

MIC for object detection is based on the following open-source projects. 
We thank their authors for making the source code publicly available.

* [sa-da-faster](https://github.com/yuhuayc/sa-da-faster)
* [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
