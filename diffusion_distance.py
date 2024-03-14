import argparse
import os

import numpy as np
from scipy.io import loadmat, savemat
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_name', type=str, default='t04_v01_s00_r04_VaryingAltitudes_A01_edges_all.mat')
    parser.add_argument('--order', type=float, default=1.0)
    parser.add_argument('--thres', type=float, default=0.65)
    parser.add_argument('--t', type=float, default=1.0)
    args = parser.parse_args()
    return args


def diffusion_distance(A, order, thres, t):
    '''
    This function calculate the diffusion distance.

    Parameters:
        A: np.array
            [num_imgs, num_imgs] array with scores between 0 and 1, indicating the scores between each image pair
        order: float
        thres: float
        t: float

    Return:
        distMat: np.array
            [num_imgs, num_imgs] array with diffusion distance
    '''
    num_imgs = A.shape[0]
    rows, cols = np.where(A > thres)
    vals = A[rows, cols]
    vals = np.power(vals, order)
    A = csr_matrix((vals, (rows, cols)), shape=(num_imgs, num_imgs))
    d = np.array(A.sum(axis=0)).flatten()
    L = csr_matrix(np.diag(d)) - A
    lambdas, U = eigh(L.todense())
    Lambdas = np.diag(np.exp(-t * lambdas))
    K = np.dot(U, np.dot(Lambdas, U.T))
    distMat = np.zeros((num_imgs, num_imgs))
    for i in range(num_imgs):
        for j in range(num_imgs):
            distMat[i, j] = K[i, i] + K[j, j] - 2 * K[i, j]
    return distMat


def assemble_A(prob, edges):
    num_edges = edges.shape[0]
    delta = np.sqrt(1 + 8 * num_edges)
    num_imgs = int((1 + delta) / 2)
    A = np.zeros((num_imgs, num_imgs))
    for p, (i, j) in zip(prob, edges):
        A[i][j] = A[j][i] = p
    return A


def save_output(args, A, distMat):
    base_name = args.in_name[:-len('.mat')]
    if base_name.endswith('_edges_all'):
        base_name = base_name[:-len('_edges_all')]
    os.makedirs(base_name, exist_ok=True)
    # save Doppelgangers results
    out_name = os.path.join(base_name, 'Doppelgangers.txt')
    np.savetxt(out_name, A)
    # save diffusion distance
    out_name = os.path.join(base_name, 'DiffusionDistance.txt')
    np.savetxt(out_name, distMat)


def main():
    args = parse_args()
    in_mat = loadmat(args.in_name)
    prob = in_mat['prob'][0]
    edges = in_mat['edges'] - 1 # conver from 1-based to 0-based indices
    A = assemble_A(prob, edges)
    distMat = diffusion_distance(A, args.order, args.thres, args.t)
    save_output(args, A, distMat)


if __name__ == '__main__':
    main()
