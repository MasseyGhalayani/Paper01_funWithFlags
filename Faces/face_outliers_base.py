# experiment meant to mirror: http://www.vision.jhu.edu/assets/TsakirisDPCPICCV15.pdf


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

# WPCA-L2 Code
#from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8119535&casa_token=RO14HRIjf5UAAAAA:FXvg29pDDWh7AT6-VEj0ERFlHY4wM6nYUejcziYqt8y3yTTGcoQauH8Owo8a0riXa9Wc4BAogg&tag=1
def WPCA_L2(X, k, init_seed = 123, eps_wt = 1e-8 , eps: float = 1e-10,  max_iter: int = 1000):

    n,p = X.shape

    U,_,_ = np.linalg.svd(X)
    W = U[:,:k]

    dJ = 1
    J = 0
    i=1

    while i < max_iter and dJ > eps*J:

        W0 = W.copy()
        J = np.sum(np.linalg.norm(X.T - X.T @ W0 @ W0.T, axis = 1))
    
        D = np.zeros((p,p))
        for i in range(p):
            x = X[:,[i]]
            wt = np.linalg.norm(x - W0 @ W0.T @ x)
            wt = np.max([wt,eps_wt])
            D[i,i] = 1/wt
        
        C = X @ D @ X.T

        _, Wall = eigh(C)

        W = Wall[:,-k:]

        Bj = np.sum(np.linalg.norm(X.T - X.T @ W@ W.T, axis = 1))
        

        dJ = J - Bj
        i+=1


    return W


#DPCP-IRLS Code

def dpcp_irls(X: np.array, k: int, eps: float = 1e-10, delta: float = 1e-8, max_iter: int = 100):
    
    n_pts = X.shape[1]

    U,_,_ = np.linalg.svd(X)
    B0 = U[:,-k:]

    J = 0
    dJ = 1
    i = 0
    while i < max_iter and dJ > eps*J:
            
        J = np.sum(np.linalg.norm(X.T @ B0, axis = 1))
        
        wts = [np.sqrt(1/np.max([delta, np.linalg.norm(B0.T @ X[:,[i]])])) for i in range(n_pts)]
        W = np.diag(np.hstack(wts))

        U,_,_ = np.linalg.svd(X @ W)

        B = U[:,-k:]

        Bj = np.sum(np.linalg.norm(X.T @ B, axis = 1))

        dJ = J - Bj

        B0 = B
        i+= 1
    
    return B0



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

                np.save(f'./results_all/labels/lbl_{n_out}_t{trial}.npy', labels)

                X_raw = np.vstack([inliers, outliers])

                pca = PCA(n_components = 50)
                X = pca.fit_transform(X_raw)

                column_means = np.mean(X, axis=0)
                Xcenter = X - column_means

                Xunit = np.vstack([x/np.linalg.norm(x) for x in X])

                labels = np.array([0]*n_in+[1]*n_out)


                Wwpca = WPCA_L2(Xcenter.T, 9)
                pca_errs = []
                for i in range(len(X)):   
                    x = Xcenter[[i],:]
                    pca_errs.append(np.linalg.norm(x @ Wwpca @ Wwpca.T  - x))
                    
                pca_preds = np.array(pca_errs)
                pca_preds = pca_preds/np.max(pca_preds)

                np.save(f'./results_all/wpca_base/preds_{face_id}_{n_out}_t{trial}.npy', pca_preds)


                Wdpcp = dpcp_irls(Xunit.T, 41)
                pca_errs = []
                for i in range(len(X)):   
                    x = Xunit[[i],:]
                    pca_errs.append(np.linalg.norm(x @ Wdpcp))

                pca_preds = np.array(pca_errs)
                pca_preds = pca_preds/np.max(pca_preds)

                np.save(f'./results_all/dpcp_base/preds_{face_id}_{n_out}_t{trial}.npy', pca_preds)

                    
                print('--------------------------------')

        print('--------------------------------')
        print('--------------------------------')
        print(f'TRIAL {trial} DONE')


