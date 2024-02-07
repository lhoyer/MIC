from tqdm import tqdm
from PIL import Image
import h5py
import os
import numpy as np
import cv2 
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/processed_data/'
dataset_folder = 'abide/caltech/'
tgtpath = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/brain/'


BRAIN_PATHS = {
    'hcp1': {
        'train': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5',
        'val': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        'test': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'hcp2': {
        'train': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5',
        'val': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        'test': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'abide_caltech': {
        'train': 'abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_10.hdf5',
        'val': 'abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_10_to_15.hdf5',
        'test': 'abide/caltech/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_16_to_36.hdf5'
        },
    'abide_stanford': {
        'train': 'abide/stanford/data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_0_to_10.hdf5',
        'val': 'abide/stanford/data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_10_to_15.hdf5',
        'test': 'abide/stanford/data_T1_2d_size_256_256_depth_132_res_0.7_0.7_from_16_to_36.hdf5'
        },
}

PALETTE = [[153, 153, 153], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70]]

dataset = 'hcp1'
split = 'train'


def convert_3d(masks):
    masks_3d = np.zeros([256, 256, 3])
    for i in range(256):
        for j in range(256):
            l = masks[i, j]
            masks_3d[i, j, :] = np.array(PALETTE[l])
    return Image.fromarray(np.uint8(masks_3d)).convert('RGB')


for dataset in ['hcp1']:
    print(dataset)
#     for split in ['test', 'train', 'val']:
    for split in ['test']:        
        dataset_folder = BRAIN_PATHS[dataset][split]

        src_file = f'{path}/{dataset_folder}'
        tgt_folder_imgs = f'{tgtpath}/{dataset}/images/{split}/'
        tgt_folder_labels = f'{tgtpath}/{dataset}/labels/{split}/'

        os.makedirs(tgt_folder_imgs, exist_ok=True) 
        os.makedirs(tgt_folder_labels, exist_ok=True) 
        
        h5_fh = h5py.File(f'{path}/{dataset_folder}', 'r')

        n_images = len(h5_fh['images'])
        print(dataset, split, n_images)
        for i_img in tqdm(range(n_images)):
            img = h5_fh['images'][i_img]
            masks = h5_fh['labels'][i_img]
            img_pil = Image.fromarray(np.uint8(img*255))#.convert('RGB')

            masks_pil_3d = convert_3d(masks)            
            masks_pil = Image.fromarray(masks)
            
            img_pil.save(f"{tgt_folder_imgs}/{i_img:04d}.png","PNG")
#             print(i_img, f"{tgt_folder_labels}/{i_img:04d}.png")
            masks_pil_3d.save(f"{tgt_folder_labels}/{i_img:04d}.png","PNG")
            masks_pil.save(f"{tgt_folder_labels}/{i_img:04d}_labelTrainIds.png","PNG")


