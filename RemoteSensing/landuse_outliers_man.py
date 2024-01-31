
from matplotlib import pyplot as plt

import numpy as np
from scipy.linalg import svd, eigh

import pandas as pd

import os

import cv2

import random

from sklearn.decomposition import PCA


import sys
sys.path.append('../scripts/')

import flag_dimensionality_reduction as fdr

import time

def get_data(class_name, n_pts):
    data = []
    for i in np.arange(n_pts):
        if i < 10:
            im_num = f'0{i}'
        else:
            im_num = str(i)
        image = plt.imread(f'./UCMerced_LandUse/Images/{class_name}/{class_name}{im_num}.tif')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small_gray_image = cv2.resize(gray_image, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
        data.append(small_gray_image.flatten())

    return data

def get_outliers(class_name, n_pts, seed):
    np.random.seed(seed)
    data = []
    for i in np.random.choice(100, n_pts, replace = False):
        if i < 10:
            im_num = f'0{i}'
        else:
            im_num = str(i)
        image = plt.imread(f'./UCMerced_LandUse/Images/{class_name}/{class_name}{im_num}.tif')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small_gray_image = cv2.resize(gray_image, dsize=(100,100), interpolation=cv2.INTER_CUBIC)
        data.append(small_gray_image.flatten())

    return data

def man_RPCA(Y: np.array, n_pcs: int = 10):
    '''
        trying to code alg1 in python
        https://jmlr.csail.mit.edu/papers/volume19/17-473/17-473.pdf
    '''
    

    pca = PCA(n_components = n_pcs)
    PCs = pca.fit_transform(Y.T)
    L0 = pca.inverse_transform(PCs).T

    n,p = L0.shape

    gamma = .1
    eta = .7

    row_idx = int(np.ceil((p-1)*(1-gamma)))
    col_idx = int(np.ceil((n-1)*(1-gamma)))
    

    diff = 1
    while diff  > 1e-10:

        U, S, V = np.linalg.svd(L0)

        L0mY = L0 - Y.T
        D = np.zeros(L0.shape)
        for i in range(n):
            sorted_row = np.sort(np.abs(L0mY[i,:]))
            row_thresh = sorted_row[row_idx]
            for j in range(p):
                sorted_col = np.sort(np.abs(L0mY[:,j]))
                col_thresh = sorted_col[col_idx]
                if L0mY[i,j] > row_thresh and L0mY[i,j] > col_thresh:
                    D[i,j] = 0
                else:
                    D[i,j] = L0mY[i,j]

        Pu = U @ U.T
        Pv = V @ V.T
        Omega = Pu @ D + D @ Pv - Pu @ D @ Pv

        L1 = L0 - eta*Omega

        diff = np.linalg.norm(L1 - L0)
        # print(np.linalg.norm(L1, ord = 'nuc') + np.linalg.norm(Xcenter - L1))
        # print(.5*np.linalg.norm(D,ord = 'fro')**2)

        L0 = L1.copy()

    return L1

if __name__ == '__main__':

    
    n_in = 100
    n_outs = [20,40,60]

    times = []

    inliers = get_data('runway', n_in)

    for trial in range(5):
        print(f'trial {trial}')

        for n_out  in n_outs:
            

            outliers = get_outliers('mobilehomepark', n_out, trial)

            X_raw = np.vstack([inliers, outliers])

            pca = PCA(n_components = 50)
            X = pca.fit_transform(X_raw)

            column_means = np.mean(X, axis=0)
            Xcenter = X - column_means

            labels = np.array([0]*n_in+[1]*n_out)

            Xhat = man_RPCA(Xcenter.T, n_pcs = 10)
            pca_errs = []
            for i in range(len(X)):   
                pca_errs.append(np.linalg.norm(Xhat[:,i]  - Xcenter[i,:]))
                
            pca_preds = np.array(pca_errs)
            pca_preds = pca_preds/np.max(pca_preds)

            np.save(f'./results/rpca_man/preds_{n_out}_t{trial}.npy', pca_preds)







