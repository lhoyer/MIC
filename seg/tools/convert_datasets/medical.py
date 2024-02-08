import h5py
from collections import Counter
import pandas as pd
from tqdm import tqdm
import json
import os.path as osp

# path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/processed_data/'
# path = '/usr/bmicnas02/data-biwi-01/klanna_data/results/generative_segmentation/data/preproc_data/'
DATASET_PATHS = {
    'hcp1': {
        'train': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_20.hdf5',
        'val': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        'test': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'hcp1_full': {
        'train': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_1040.hdf5',
        'val': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        # 'test': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
        },
    'hcp2_full': {
        'train': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_0_to_1040.hdf5',
        'val': 'hcp/data_T2_2d_size_256_256_depth_256_res_0.7_0.7_from_20_to_25.hdf5',
        # 'test': 'hcp/data_T1_2d_size_256_256_depth_256_res_0.7_0.7_from_50_to_70.hdf5'
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
    'nci': {
        'all': 'nci/data_2d_size_256_256_res_0.625_0.625_cv_fold_1.hdf5',
        },
    'pirad_erc': {
        'train': 'pirad_erc/data_2d_from_40_to_68_size_256_256_res_0.625_0.625_ek.hdf5',
        'test': 'pirad_erc/data_2d_from_0_to_20_size_256_256_res_0.625_0.625_ek.hdf5',
        'val': 'pirad_erc/data_2d_from_20_to_40_size_256_256_res_0.625_0.625_ek.hdf5'
        },
    'acdc': {
        'all': 'acdc/data_2D_size_256_256_res_1.33_1.33_cv_fold_1.hdf5',
        },
    'rvsc': {
        'all': 'rvsc/data_2D_size_256_256_res_1.33_1.33_cv_fold_1.hdf5',
        },
    'umc': {
        'train': 'umc/data_umc_2d_size_256_256_depth_48_res_1_1_3_from_0_to_10.hdf5',
        'val': 'umc/data_umc_2d_size_256_256_depth_48_res_1_1_3_from_10_to_15.hdf5',
        'test': 'umc/data_umc_2d_size_256_256_depth_48_res_1_1_3_from_15_to_20.hdf5'
    },
    'nuhs': {
        'train': 'nuhs/data_nuhs_2d_size_256_256_depth_48_res_1_1_3_from_0_to_10.hdf5',
        'val': 'nuhs/data_nuhs_2d_size_256_256_depth_48_res_1_1_3_from_10_to_15.hdf5',
        'test': 'nuhs/data_nuhs_2d_size_256_256_depth_48_res_1_1_3_from_15_to_20.hdf5'
    },
    'vu': {
        'train': 'vu/data_vu_2d_size_256_256_depth_48_res_1_1_3_from_0_to_10.hdf5',
        'val': 'vu/data_vu_2d_size_256_256_depth_48_res_1_1_3_from_10_to_15.hdf5',
        'test': 'vu/data_vu_2d_size_256_256_depth_48_res_1_1_3_from_15_to_20.hdf5'
    }

}

# path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/wmh_miccai/'
# dataset = 'nuhs'
# tgtpath = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/brain/'

# path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/prostate/'
path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/processed_data/'
# dataset = 'nci'
# split = 'all'

dataset = 'pirad_erc'
split = 'train'
tgtpath = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/prostate/'


def get_freq(targets):
    counter_classes = dict(Counter(targets))
    counter_classes_str = {}
    for k in counter_classes:
        counter_classes_str[str(k)] = counter_classes[k]
    return counter_classes_str

def get_masks(dataset='hcp1'):
    dataset_folder = DATASET_PATHS[dataset][split]
    
    h5_fh = h5py.File(f'{path}/{dataset_folder}', 'r')    
    # print(h5_fh['masks_train'].shape)
    # uncomment for nci
    # masks = h5_fh['masks_train']
    masks = h5_fh['labels']
    
    return masks

def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


masks = get_masks(dataset)

n_images = masks.shape[0]
pbar = tqdm(range(n_images))
freq_list = []
for i in pbar:
    freq_dict = get_freq(list(masks[i].reshape(-1)))
    freq_dict['file'] = f'{tgtpath}/{dataset}/labels/{split}/{i:04d}_labelTrainIds.png'
    freq_list.append(freq_dict)

save_class_stats(f'{tgtpath}/{dataset}/', freq_list)