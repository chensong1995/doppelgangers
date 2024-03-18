'''
This file is extracted from loftr_matches.py in the original Doppelgangers
implementation to make sure the pairs_dataset.py can reuse the functions.
'''

import cv2
import numpy as np
from PIL import Image, ImageOps


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new


def read_image(img_pth, img_size, df, padding):
    if str(img_pth).endswith('gif'):
        pil_image = ImageOps.grayscale(Image.open(str(img_pth)))
        img_raw = np.array(pil_image)
    else:
        img_raw = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)

    w, h = img_raw.shape[1], img_raw.shape[0]
    w_new, h_new = get_resized_wh(w, h, img_size)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)

    if padding:  # padding
        pad_to = max(h_new, w_new)    
        mask = np.zeros((1, pad_to, pad_to), dtype=bool)
        mask[:, :h_new, :w_new] = True
        mask = mask[:, ::8, ::8]
    
    image = cv2.resize(img_raw, (w_new, h_new))
    pad_image = np.zeros((1, 1, pad_to, pad_to), dtype=np.float32)
    pad_image[0, 0, :h_new, :w_new] = image / 255.0

    return pad_image, mask
