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

def load_inliers(face_num):
    yale_faces = []

    base_dir  = './data/CroppedYale/'
    subject_id = os.listdir(base_dir)[face_num]
    if '.DS' not in subject_id:
        subject_dir = os.path.join(base_dir, subject_id)
        for file_id in os.listdir(subject_dir):
            if ('.pgm' in file_id) and ('Ambient' not in file_id):
                file_dir = os.path.join(subject_dir, file_id)
                image = plt.imread(file_dir)
                small_image = cv2.resize(image, dsize=(30,30), interpolation=cv2.INTER_CUBIC)
                yale_faces.append(small_image.flatten())

    yale_faces = np.vstack(yale_faces)
    n_in = len(yale_faces)

    return yale_faces, n_in

def load_outliers(n_out):
    base_dir = './data/Caltech101/101_ObjectCategories/'

    random.seed(123)
    caltech_images = []
    for i in range(n_out):
        # dir_name = random.choice(os.listdir(base_dir))
        dir_name = os.listdir(base_dir)[3]
        dir_path = os.path.join(base_dir,dir_name)
        image_name = random.choice(os.listdir(dir_path))
        file_dir = os.path.join(dir_path, image_name)
        image = plt.imread(file_dir)
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        small_gray_image = cv2.resize(gray_image, dsize=(30,30), interpolation=cv2.INTER_CUBIC)
        caltech_images.append(small_gray_image.flatten())
        

    caltech_images = np.vstack(caltech_images)

    return caltech_images



if __name__ == '__main__':

    n_outs = [32, 64, 128]
    fl_types = [[3],[1,3],[2,3],[1,2,3]]

    for face_num in np.arange(12,40):

        times = []

        inliers, n_in = load_inliers(face_num)

        for n_out  in n_outs:

            start = time.time()

            outliers = load_outliers(n_out)

            X = np.vstack([inliers, outliers])

            column_means = np.mean(X, axis=0)
            Xcenter = X - column_means

            labels = np.array([0]*n_in+[1]*n_out)

            pca = PCA(n_components = 3)
            pca.fit(X)
            Wpca = pca.components_.T
            pca_errs = []
            for i in range(len(X)):   
                x = Xcenter[[i],:]
                pca_errs.append(np.linalg.norm(x @ Wpca @ Wpca.T  - x))
                
            pca_preds = np.array(pca_errs)
            pca_preds = pca_preds/np.max(pca_preds)

            np.save(f'./results/pca/preds_{face_num}_{n_out}.npy', pca_preds)

            for fl_type in fl_types:

                Wfpca = fdr.flag_robust_pca(Xcenter.T, fl_type, 'wpca', verbose = False, 
                                            return_all = False, max_iters = 200, init = 'rand')

                fpca_errs = []
                for i in range(len(X)):   
                    x = Xcenter[[i],:]
                    fpca_errs.append(np.linalg.norm(x @ Wfpca @ Wfpca.T  - x))
                    
                fpca_preds = np.array(fpca_errs)
                fpca_preds = fpca_preds/np.max(fpca_preds)

                np.save(f'./results/wpca/preds_{face_num}_{fl_type}_{n_out}.npy', fpca_preds)
                
                print(f'{fl_type} {n_out}')

            times.append(time.time() - start)
                
            print()
            print()
            print('--------------------------------')


        times = np.array(times)
        np.save(f'results/wpca/{face_num}_times.npy', times)


