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

def load_inliers(ii):
    yale_faces = []

    base_dir  = './data/CroppedYale/'
    subject_id = sorted(os.listdir(base_dir))[ii]
    if '.DS' not in subject_id:
        subject_dir = os.path.join(base_dir, subject_id)
        for file_id in sorted(os.listdir(subject_dir)):
            if ('.pgm' in file_id) and ('Ambient' not in file_id):
                file_dir = os.path.join(subject_dir, file_id)
                image = plt.imread(file_dir)
                small_image = cv2.resize(image, dsize=(52,48), interpolation=cv2.INTER_CUBIC)
                yale_faces.append(small_image.flatten())

    yale_faces = np.vstack(yale_faces)
    n_in = len(yale_faces)

    return yale_faces, n_in, subject_id[-2:]

def load_outliers(n_out, trial):
    base_dir = './data/Caltech101/101_ObjectCategories/'

    random.seed(trial)
    caltech_images = []
    for _ in range(n_out):
        dir_name = random.choice(sorted(os.listdir(base_dir)))
        # dir_name = os.listdir(base_dir)[3]
        dir_path = os.path.join(base_dir,dir_name)
        image_name = random.choice(sorted(os.listdir(dir_path)))
        file_dir = os.path.join(dir_path, image_name)
        image = plt.imread(file_dir)
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        small_gray_image = cv2.resize(gray_image, dsize=(52,48), interpolation=cv2.INTER_CUBIC)
        caltech_images.append(small_gray_image.flatten())
        

    caltech_images = np.vstack(caltech_images)

    return caltech_images

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

        L0mY = L0 - Xcenter.T
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

    n_outs = [32, 64, 128]

    for trial in range(5):

        for face_num in np.arange(1,39):

            times = []

            inliers, n_in, face_id = load_inliers(face_num)
            print(f'starting B{face_id}')
            print('.')
        
            for n_out  in n_outs:

                outliers = load_outliers(n_out, trial)

                labels = np.array([0]*n_in + [1]*n_out)

                X_raw = np.vstack([inliers, outliers])

                pca = PCA(n_components = 50)
                X = pca.fit_transform(X_raw)

                column_means = np.mean(X, axis=0)
                Xcenter = X - column_means

                Xunit = np.vstack([x/np.linalg.norm(x) for x in X])

                labels = np.array([0]*n_in+[1]*n_out)

                Xhat = man_RPCA(Xcenter.T, n_pcs = 10)
                pca_errs = []
                for i in range(len(X)):   
                    pca_errs.append(np.linalg.norm(Xhat[:,i]  - Xcenter[i,:]))
                    
                pca_preds = np.array(pca_errs)
                pca_preds = pca_preds/np.max(pca_preds)

                np.save(f'./results_all/rpca_man/preds_{face_id}_{n_out}_t{trial}.npy', pca_preds)

                for pca_type in ['wpca', 'dpcp']:
                    start = time.time()

                    if pca_type == 'dpcp':
                        Wfpca = fdr.flag_robust_pca(Xunit.T, [11,41], pca_type, verbose = False, 
                                                    return_all = False, max_iters = 200, init = 'rand')

                        fpca_errs = []
                        for i in range(len(X)):   
                            x = Xunit[[i],:]
                            fpca_errs.append(np.linalg.norm(x @ Wfpca))

                    else:
                        Wfpca = fdr.flag_robust_pca(Xcenter.T, [3,9], pca_type, verbose = False, 
                                                    return_all = False, max_iters = 200, init = 'rand')

                        fpca_errs = []
                        for i in range(len(X)):   
                            x = Xcenter[[i],:]
                            fpca_errs.append(np.linalg.norm(x @ Wfpca @ Wfpca.T  - x))

                    np.save(f'./results_all/{pca_type}1/w_{face_id}_{n_out}_t{trial}.npy', Wfpca)
                        
                    fpca_preds = np.array(fpca_errs)
                    fpca_preds = fpca_preds/np.max(fpca_preds)

                    np.save(f'./results_all/{pca_type}1/preds_{face_id}_{n_out}_t{trial}.npy', fpca_preds)
                    
                    print(f'{pca_type} {n_out}')
                    run_time = time.time()-start
                    print(f'time = {run_time}')
                    print('----------')
                    
                print('--------------------------------')

        print('--------------------------------')
        print('--------------------------------')
        print(f'TRIAL {trial} DONE')


