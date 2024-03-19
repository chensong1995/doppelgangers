import torch
import numpy as np
import os.path as osp
import tqdm

from .input_utils import read_image
from .pairs_dataset import PairsDataset
from ..third_party.loftr import LoFTR, default_cfg


def save_loftr_matches(data_path, pair_path, output_path, model_weight_path="weights/outdoor_ds.ckpt"):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_weight_path)['state_dict'])
    matcher = matcher.eval().cuda()

    pairs_info = np.load(pair_path, allow_pickle=True)
    img_size = 1024
    df = 8
    padding = True

    for idx in tqdm.tqdm(range(pairs_info.shape[0])):
        output_dir = osp.join(output_path, f'loftr_match/{idx}.npy')
        if osp.exists(output_dir):
            continue
        name0, name1, _, _, _ = pairs_info[idx]

        img0_pth = osp.join(data_path, name0)
        img1_pth = osp.join(data_path, name1)
        img0_raw, mask0 = read_image(img0_pth, img_size, df, padding)
        img1_raw, mask1 = read_image(img1_pth, img_size, df, padding)        
        img0 = torch.from_numpy(img0_raw).cuda()
        img1 = torch.from_numpy(img1_raw).cuda()
        mask0 = torch.from_numpy(mask0).cuda()
        mask1 = torch.from_numpy(mask1).cuda()
        batch = {'image0': img0, 'image1': img1, 'mask0': mask0, 'mask1':mask1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            np.save(output_dir, {"kpt0": mkpts0, "kpt1": mkpts1, "conf": mconf})


def save_loftr_matches_parallel(
    data_path,
    pair_path,
    output_path,
    model_weight_path='weights/outdoor_ds.ckpt',
    batch_size=8,
    num_workers=4,
    img_size=1024,
):
    '''
    Save the LoFTR matches in parallel. This is the parallel implementation of
    save_loftr_matches().

    Parameters:
        data_path: str
            Path to the directory containing all the images of interest.
        pair_path: str
            Path to the .npy file created by create_pair_list().
        output_path: str
            Path to the directory where the output .npy files will be stored.
        model_weight_path: str
            Path to the LoFTR weights.
        batch_size: int
            Total batch size across all GPUs.
        num_workers: int
            Number of CPU workers to load the pairs.
        img_size: int
            Side length of the image after resizing.
    '''
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_weight_path)['state_dict'])
    matcher = torch.nn.DataParallel(matcher.eval()).cuda()

    dataset = PairsDataset(data_path, pair_path, img_size=img_size)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            idx = batch['idx'].cpu().numpy()
            batch_size = idx.shape[0]

            num_processed = 0
            for i in range(len(idx)):
                output_dir = osp.join(output_path, f'loftr_match/{idx[i]}.npy')
                if osp.exists(output_dir):
                    num_processed += 1
            if num_processed == batch_size:
                continue

            batch = matcher(batch)

            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()

            for i in range(len(idx)):
                output_dir = osp.join(output_path, f'loftr_match/{idx[i]}.npy')
                np.save(
                    output_dir,
                    {
                        'kpt0': np.expand_dims(mkpts0[i], axis=0),
                        'kpt1': np.expand_dims(mkpts1[i], axis=0),
                        'conf': np.expand_dims(mconf[i], axis=0)
                    }
                )


