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

    
    n_in = 100
    n_outs = [20,40,60]
    fl_types = [[1,10]]

    times = []

    inliers = get_data('runway', n_in)

    for trial in range(5):

        for n_out  in n_outs:

            

            outliers = get_outliers('mobilehomepark', n_out, trial)

            X_raw = np.vstack([inliers, outliers])

            pca = PCA(n_components = 10)
            pca.fit(X_raw)
            Wpca = pca.components_.T

            column_means = np.mean(X_raw, axis=0)
            X_rawcenter = X_raw - column_means
            
            pca_errs = []
            for i in range(len(X_raw)):   
                x = X_rawcenter[[i],:]
                pca_errs.append(np.linalg.norm(x @ Wpca @ Wpca.T  - x))
                
            pca_preds = np.array(pca_errs)
            pca_preds = pca_preds/np.max(pca_preds)

            np.save(f'./results/pca_all/preds_{n_out}_t{trial}.npy', pca_preds)



            pca = PCA(n_components = 50)
            X = pca.fit_transform(X_raw)

            column_means = np.mean(X, axis=0)
            Xcenter = X - column_means

            Xunit = np.vstack([x/np.linalg.norm(x) for x in X])


            labels = np.array([0]*n_in+[1]*n_out)

            pca = PCA(n_components = 10)
            pca.fit(X)
            Wpca = pca.components_.T
            pca_errs = []
            for i in range(len(X)):   
                x = Xcenter[[i],:]
                pca_errs.append(np.linalg.norm(x @ Wpca @ Wpca.T  - x))
                
            pca_preds = np.array(pca_errs)
            pca_preds = pca_preds/np.max(pca_preds)

            np.save(f'./results_sanity/pca/preds_{n_out}_t{trial}.npy', pca_preds)

            Xhat = man_RPCA(Xcenter.T, n_pcs = 10)
            pca_errs = []
            for i in range(len(X)):   
                pca_errs.append(np.linalg.norm(Xhat[:,i]  - Xcenter[i,:]))
                
            pca_preds = np.array(pca_errs)
            pca_preds = pca_preds/np.max(pca_preds)

            np.save(f'./results_sanity/rpca_man/preds_{n_out}_t{trial}.npy', pca_preds)

            Wwpca = WPCA_L2(Xcenter.T, 10)
            pca_errs = []
            for i in range(len(X)):   
                x = Xcenter[[i],:]
                pca_errs.append(np.linalg.norm(x @ Wwpca @ Wwpca.T  - x))
                
            pca_preds = np.array(pca_errs)
            pca_preds = pca_preds/np.max(pca_preds)

            np.save(f'./results_sanity/wpca_base/preds_{n_out}_t{trial}.npy', pca_preds)


            Wdpcp = dpcp_irls(Xunit.T, 40)
            pca_errs = []
            for i in range(len(X)):   
                x = Xunit[[i],:]
                pca_errs.append(np.linalg.norm(x @ Wdpcp))

            pca_preds = np.array(pca_errs)
            pca_preds = pca_preds/np.max(pca_preds)

            np.save(f'./results_sanity/dpcp_base/preds_{n_out}_t{trial}.npy', pca_preds)

            for fl_type in fl_types:


                for pca_type in ['rpca', 'wpca', 'dpcp']:

                    if pca_type == 'dpcp':
                        if fl_type == [1,10]:
                            dfl_type = [1,40]
                        else:
                            dfl_type = [40]
                        Wfpca = fdr.flag_robust_pca(Xunit.T, dfl_type, pca_type, verbose = True, 
                                                    return_all = False, max_iters = 200, init = 'rand')

                        fpca_errs = []
                        for i in range(len(X)):   
                            x = Xunit[[i],:]
                            fpca_errs.append(np.linalg.norm(x @ Wfpca))

                    else:
                        Wfpca = fdr.flag_robust_pca(Xcenter.T, fl_type, pca_type, verbose = False, 
                                                    return_all = False, max_iters = 200, init = 'rand')

                        fpca_errs = []
                        for i in range(len(X)):   
                            x = Xcenter[[i],:]
                            fpca_errs.append(np.linalg.norm(x @ Wfpca @ Wfpca.T  - x))
                            
                    fpca_preds = np.array(fpca_errs)
                    fpca_preds = fpca_preds/np.max(fpca_preds)

                    np.save(f'./results_sanity/{pca_type}/preds_{fl_type}_{n_out}_t{trial}.npy', fpca_preds)
                    
                    print(f'{pca_type} {fl_type} {n_out}')

                    
        print()
        print()
        print('--------------------------------')





