# MIC for Domain-Adaptive Image Classification

## Getting started

### Installation

For this project, we used python 3.8.5. We recommend setting up a new virtual environment:

```shell
python -m venv ~/venv/mic-cls
source ~/venv/mic-cls/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt
```

We use Weights and Biases ([wandb](https://wandb.ai/site)) to track our experiments and results. 
For that purpose, please create a new project `MIC` with your account.

### Datasets

The datasets used in the repository can be downloaded from the following links:

* [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
* [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public) (under classification track)

The datasets are automatically downloaded to the ```examples/data/``` folder if it is not available.

## Training

For experiments in our paper, we use a script to automatically
generate the different configurations and train them:

```shell
python run_experiments.py --exp <ID>
```

More information about the available experiments and their assigned IDs, can be
found in [experiments.py](experiments.py).

The experiment progress is logged on Weights&Biases.

For VisDA-2017, the mean over the class accuracies is reported. This value is denoted
as 'mean correct' in the logs as explained in https://github.com/val-iisc/SDAT/issues/1. 

## Where to find MIC in the code?

The most relevant files for MIC are:

* [experiment.py](experiments.py):
  Definition of the experiment configurations in our paper.
* [examples/cdan_mcc_sdat_masking.py](examples/cdan_mcc_sdat_masking.py):
  Training script for UDA with MIC(SDAT).
* [dalib/modules/masking.py](dalib/modules/masking.py):
  Implementation of MIC.
* [dalib/modules/teacher.py](dalib/modules/teacher.py):
  Implementation of the EMA teacher.

## Acknowledgements

MIC for classification is based on the following open-source projects. 
We thank their authors for making the source code publicly available.

* [SDAT](https://github.com/val-iisc/SDAT)
* [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library)

