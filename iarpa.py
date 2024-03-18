'''
Description:
    This script calcluates the matching scores between input image pairs.

    * The input argument --image_dir points to a directory containing all the images
    of interest. Suppose the directory contains n images, the matching score will be
    calculated for all n * (n - 1) / 2 pairs. For each pair, the matching score is a
    scalar between 0 and 1. The higher the score is, the more likely the pair is good.

    * The input argument --temp_dir specifies the directory to store intermediate
    results. When the script finishes, the directory will contain edges_all.mat.

    * The input argument --batch_size specifies thee total batch size in all GPUs.

    * The input argument --num_workers specifies the number of CPU workers in the
    dataloader.

    * The input argument --img_size specifies the side length of the image after
    resizing in LoFTR inference.

    * The output of the script is stored in edges_all.mat as a MATLAB data file. The
    file includes three objects.
        - `prob` is an array with a length of n * (n - 1) / 2, indicating the
        matching scores (in other words, the probabilities) for each pair. Each
        element in the array is a 2-tuple, representing the raw values before passing
        to the softmax activation function to extract the probabilities.
        - `edges` is an array of two-tuples with a length of n * (n - 1) / 2,
        indicating the pair of image ids for each corresponding entry in `prob`.
        - `xform` is an array of 3x3 estimated transformation matrices in 2D with a
        length of n * (n - 1) / 2. Each 3x3 matriX transforms the first image into
        the second image in the corresponding pair.

Usage:
    python iarpa.py \
        --image_dir <dir_containing_all_jpg_images> \
        --temp_dir <temp_dir> \
        --batch_size <batch_size> \
        --num_workers <num_workers> \
        --img_size <img_size>
'''

import argparse
import os
import subprocess

import numpy as np
from scipy.io import loadmat, savemat
from scipy.special import softmax
import torch
import yaml

from doppelgangers.utils.database import image_ids_to_pair_id
from doppelgangers.utils.loftr_matches import save_loftr_matches_parallel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='../data_more/t04_v01_s00_r04_VaryingAltitudes_A01'
    )
    parser.add_argument('--temp_dir', type=str, default='temp')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=1024)
    args = parser.parse_args()
    return args


def create_pair_list(args, pairs):
    # some image names do not end with .jpg. They end with png or other 3-letter (?)
    # extensions.
    image_id_to_image_name = {}
    for image_name in os.listdir(args.image_dir):
        image_id = int(image_name[len('image_'):-len('.jpg')])
        image_id_to_image_name[image_id] = image_name
    # modified from doppelgangers/utils/process_database.py:create_image_pair_list
    pair_list = []
    for i, (image_id1, image_id2) in enumerate(pairs):
        name1 = image_id_to_image_name[image_id1]
        name2 = image_id_to_image_name[image_id2]
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        # label and num_matches are not used by save_loftr_matches
        label = 0
        num_matches = 0
        pair_list.append([name1, name2, label, num_matches, pair_id])
    pair_list = np.concatenate(pair_list, axis=0).reshape(-1, 5)
    np.save('%s/pairs_list.npy' % args.temp_dir, pair_list)
    return '%s/pairs_list.npy' % args.temp_dir


def setup_doppelgangers(args, pairs):
    # extract LoFTR matches
    loftr_matches_path = os.path.join(args.temp_dir, 'loftr_match')
    os.makedirs(loftr_matches_path, exist_ok=True)
    pair_path = create_pair_list(args, pairs)
    save_loftr_matches_parallel(
        args.image_dir,
        pair_path,
        args.temp_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    torch.cuda.empty_cache()

    # create config file
    example_config_file = 'doppelgangers/configs/test_configs/sfm_disambiguation_example.yaml'
    with open(example_config_file) as f:
        example_config = yaml.safe_load(f)

    example_config['data']['image_dir'] = args.image_dir
    example_config['data']['loftr_match_dir'] = loftr_matches_path
    example_config['data']['test']['batch_size'] = args.batch_size
    example_config['data']['test']['pair_path'] = pair_path
    example_config['data']['output_path'] = args.temp_dir

    config_path = '%s/config.yaml' % args.temp_dir
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f)
    return config_path


def load_pairs(args):
    end = len(os.listdir(args.image_dir))
    edges = np.array([(i, j) for i in range(1, end + 1) for j in range(i + 1, end + 1)])
    return edges


def check_pairs(args, pairs):
    config_path = setup_doppelgangers(args, pairs)

    command = [
        'python',
        'test_sfm_disambiguation.py', 
        config_path,
        '--pretrained',
        'weights/doppelgangers_classifier_loftr.pt'
    ]
    subprocess.run(' '.join(command), shell=True)
    torch.cuda.empty_cache()

    pair_probability_path = os.path.join(args.temp_dir, 'pair_probability_list.npy')
    # adapted from doppelgangers/utils/process_database.py:remove_doppelgangers
    pair_probability = np.load(pair_probability_path, allow_pickle=True).item()
    y_scores = np.array(pair_probability['prob']).reshape(-1, 2)
    y_scores = softmax(y_scores, axis=1)[:, 1]
    M = pair_probability['M']
    return y_scores, M


def main():
    args = parse_args()
    edges = load_pairs(args)

    probability, M = check_pairs(args, edges)

    savemat(
        os.path.join(args.temp_dir, 'edges_all.mat'),
        { 'prob': probability, 'xform': M, 'edges': edges }
    )


if __name__ == '__main__':
    main()
