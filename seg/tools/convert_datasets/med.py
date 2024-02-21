import json
import os.path as osp
from collections import Counter
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

def get_subfolders(directory):
    """
    Returns a list of names of immediate subfolders in the given directory.
    """
    return sorted([name.split('.png')[0] for name in os.listdir(directory)
            if '.png' in name])

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


def get_freq(targets):
    counter_classes = dict(Counter(targets))
    counter_classes_str = {}
    for k in counter_classes:
        counter_classes_str[str(k)] = counter_classes[k]
    return counter_classes_str

def get_class_stats(path, split = 'train'):    
    freq_list = []
    subfolders = get_subfolders(f"{path}/images/{split}")
    for idx in tqdm(subfolders):    
        masks_file_name = f"{path}/labels/{split}/{idx}_labelTrainIds.png"
        masks = Image.open(masks_file_name)
        masks = np.array(masks, dtype=np.int32)
        
        freq_dict = get_freq(list(masks.reshape(-1)))
        freq_dict['file'] = masks_file_name
        freq_list.append(freq_dict)

    save_class_stats(path, freq_list)



path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/lumbarspine/'
# get_class_stats(f'{path}/MRSpineSegV/')
get_class_stats(f'{path}/VerSe-dual/')


# path = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/brain/'
# get_class_stats(f'{path}/hcp1-dual/')
# get_class_stats(f'{path}/umc-dual/')