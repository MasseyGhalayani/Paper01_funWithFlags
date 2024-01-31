# experiment meant to mirror: http://www.vision.jhu.edu/assets/TsakirisDPCPICCV15.pdf


from matplotlib import pyplot as plt

import numpy as np

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


if __name__ == '__main__':

    
    n_in = 100
    n_outs = [20,40,60]
    fl_types = [[1,2,3,4,5,6,7,8,9,10]]

    times = []

    inliers = get_data('runway', n_in)

    for trial in range(5):

        for n_out  in n_outs:

            outliers = get_outliers('mobilehomepark', n_out, trial)

            X_raw = np.vstack([inliers, outliers])


            pca = PCA(n_components = 50)
            X = pca.fit_transform(X_raw)

            column_means = np.mean(X, axis=0)
            Xcenter = X - column_means


            labels = np.array([0]*n_in+[1]*n_out)



            for fl_type in fl_types:


                for pca_type in ['rpca']:

                    Wfpca = fdr.flag_robust_pca(Xcenter.T, fl_type, pca_type, verbose = False, 
                                                return_all = False, max_iters = 200, init = 'rand')

                    fpca_errs = []
                    for i in range(len(X)):   
                        x = Xcenter[[i],:]
                        fpca_errs.append(np.linalg.norm(x @ Wfpca @ Wfpca.T  - x))
                            
                    fpca_preds = np.array(fpca_errs)
                    fpca_preds = fpca_preds/np.max(fpca_preds)

                    np.save(f'./results/{pca_type}/preds_{fl_type}_{n_out}_t{trial}.npy', fpca_preds)
                    
                    print(f'{pca_type} {fl_type} {n_out}')

                    
        print()
        print()
        print('--------------------------------')





