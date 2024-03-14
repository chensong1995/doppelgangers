import argparse
import os
import subprocess

import numpy as np
from scipy.io import loadmat, savemat
from scipy.special import softmax
import torch
import yaml

from doppelgangers.utils.database import image_ids_to_pair_id
from doppelgangers.utils.loftr_matches import save_loftr_matches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='../data_more/t04_v01_s00_r04_VaryingAltitudes_A01'
    )
    parser.add_argument('--temp_dir', type=str, default='tempp')
    args = parser.parse_args()
    return args


def create_pair_list(args, pairs):
    # modified from doppelgangers/utils/process_database.py:create_image_pair_list
    pair_list = []
    for i, (image_id1, image_id2) in enumerate(pairs):
        name1 = 'image_%06d.jpg' % image_id1
        name2 = 'image_%06d.jpg' % image_id2
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
    save_loftr_matches(args.image_dir, pair_path, args.temp_dir)
    torch.cuda.empty_cache()

    # create config file
    example_config_file = 'doppelgangers/configs/test_configs/sfm_disambiguation_example.yaml'
    with open(example_config_file) as f:
        example_config = yaml.safe_load(f)

    example_config['data']['image_dir'] = args.image_dir
    example_config['data']['loftr_match_dir'] = loftr_matches_path
    example_config['data']['test']['pair_path'] = pair_path
    example_config['data']['output_path'] = args.temp_dir

    config_path = '%s/config.yaml' % args.temp_dir
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f)
    return config_path


def load_pairs(args):
    end = len(os.listdir(args.image_dir)) + 1
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

    args.temp_dir = os.path.join(args.temp_dir, 'edges_all')
    probability, M = check_pairs(args, edges)

    savemat(
        os.path.join(args.temp_dir, 'edges_all.mat'),
        { 'prob': probability, 'xform': M, 'edges': edges }
    )


if __name__ == '__main__':
    main()
