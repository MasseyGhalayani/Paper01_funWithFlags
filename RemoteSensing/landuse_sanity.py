
import numpy as np
import numpy.linalg as la

from matplotlib import pyplot as plt

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


import math
# import numpy.linalg
from scipy.linalg import svd

import wpca


def robust_pca(M):
    #https://github.com/nwbirnie/rpca/blob/master/rpca.py
    """ 
    Decompose a matrix into low rank and sparse components.
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    Returns L,S the low rank and sparse components respectively
    """
    L = numpy.zeros(M.shape)
    S = numpy.zeros(M.shape)
    Y = numpy.zeros(M.shape)
    # print(M.shape)
    mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    while not converged(M,L,S):
        L = svd_shrink(M - S - (mu**-1) * Y, mu)
        S = shrink(M - L + (mu**-1) * Y, lamb * mu)
        Y = Y + mu * (M - L - S)
    return L,S
    
def svd_shrink(X, tau):
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.
    The parameter tau is used as the scaling parameter to the shrink function.
    Returns the matrix obtained by computing U * shrink(s) * V where 
        U are the left singular vectors of X
        V are the right singular vectors of X
        s are the singular values as a diagonal matrix
    """
    U,s,V = numpy.linalg.svd(X, full_matrices=False)
    return numpy.dot(U, numpy.dot(numpy.diag(shrink(s, tau)), V))
    
def shrink(X, tau):
    """
    Apply the shrinkage operator the the elements of X.
    Returns V such that V[i,j] = max(abs(X[i,j]) - tau,0).
    """
    V = numpy.copy(X).reshape(X.size)
    for i in range(V.size):
        V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
        if V[i] == -0:
            V[i] = 0
    return V.reshape(X.shape)
            
def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = 0
    V = numpy.reshape(X,X.size)
    for i in range(V.size):
        accum += abs(V[i] ** 2)
    return math.sqrt(accum)

def L1Norm(X):
    """
    Evaluate the L1 norm of X
    Returns the max over the sum of each column of X
    """
    return max(numpy.sum(X,axis=0))

def converged(M,L,S):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    """
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    print(f'error = {error}')
    return error <= 10e-6

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
# RPCA Code
# translated from https://github.com/Markopoulos-Research/Efficient-L1-Norm-Principal-Component-Analysis-via-Bit-Flipping

def l1pca_sbfk(X, K, num_init, print_flag):
	# Parameters
	toler =10e-8

	# Get the dimentions of the matrix.
	dataset_matrix_size = X.shape	
	D = dataset_matrix_size[0]	# Row dimension.
	N = dataset_matrix_size[1]	# Column dimension.

	# Initialize the matrix with the SVD.
	dummy, S_x, V_x = svd(X , full_matrices = False)	# Hint: The singular values are in vector form.
	if D < N:
		V_x = V_x.transpose()
		
	X_t = np.matmul(np.diag(S_x),V_x.transpose())

	# Initialize the required matrices and vectors.
	Bprop = np.ones((N,K),dtype=float)
	nucnormmax = 0
	iterations = np.zeros((1,num_init),dtype=float)

	# For each initialization do.
	for ll in range(0, num_init):
		
		start_time = time.time()	# Start measuring execution time.

		v = np.random.randn(N,K)	# Random initialized vector.
		if ll<2:	# In the first initialization, initialize the B matrix to sign of the product of the first 
					# right singular vector of the input matrix with an all-ones matrix.
			z = np.zeros((N,1),dtype=float)
			z = V_x[:,0]
			z_x = z.reshape(N,1)
			v = np.matmul(z_x,np.ones((1,K), dtype=float))
		B = np.sign(v)	# Get a binary vector containing the signs of the elements of v.

		# Calculate the nuclear norm of X*B.
		X_temp = np.matmul(X_t,B)
		dummy1, S, dummy2 = svd(X_temp , full_matrices = False)
		nucnorm = np.sum(np.sum(np.diag(S)))
		nuckprev = nucnorm*np.ones((K,1), dtype=float)

		# While not converged bit flip.
		iter_ = 0
		while True:
			iter_ = iter_ + 1

			flag = False

			# Calculate all the possible binary vectors and all posible bit flips.
			for k in range(0, K):

				a = np.zeros((N,1), dtype=float)

				for n in range(0, N):
					B_t = B
					B_t[n,k] = -B[n,k]
					dummy1, S, dummy2 = svd(np.matmul(X_t,B), full_matrices=False)
					a[n] = sum(sum(np.diag(S)))
				
				ma = np.max(a)	# Find which binary vector and bit flips maximize the quadratic.
				if ma > nucnorm:
					nc = np.where(a == ma)
					B_t[nc[0],k] = -B_t[nc[0],k]
					nucnorm = ma

				# If the maximum quadratic is attained, stop iterating.
				if iter_ > 1 and nucnorm<nuckprev[k] + toler:
					flag = True
					break

				nuckprev[k] = nucnorm # Save the calculated nuclear norm of the current initialization.

			if flag == True:
				break

		# Find the maximum nuclear norm across all initializations.
		iterations[0,ll] = iter_
		if nucnorm > nucnormmax:
			nucnormmax = nucnorm
			Bprop = B

	# Calculate the final subspace.
	U, dummy, V = svd(np.matmul(X,Bprop), full_matrices=False)
	Uprop = U[:,0:K]
	Vprop = V[:,0:K]
	Qprop = np.matmul(Uprop,Vprop.transpose())

	end_time = time.time()	# End of execution timestamp.
	timelapse = (end_time - start_time)	# Calculate the time elapsed.

	convergence_iter = np.mean(iterations, dtype=float) # Calculate the mean iterations per initialization.
	vmax = sum(sum(abs(np.matmul(Qprop.transpose(),X))))
	
	# If print true, print execution statistics.
	if print_flag:
		print("------------------------------------------")
		print("Avg. iterations/initialization: ", (convergence_iter))
		print("Time elapsed (sec): ", (timelapse))
		print("Metric value:", vmax)
		print("------------------------------------------")

	return Qprop, Bprop, vmax


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

            # Xhat, S = robust_pca(Xcenter.T)
            # pca_errs = []
            # for i in range(len(X)):   
            #     pca_errs.append(np.linalg.norm(Xhat[:,i]  - Xcenter[i,:]))
                
            # pca_preds = np.array(pca_errs)
            # pca_preds = pca_preds/np.max(pca_preds)

            # np.save(f'./results/rpca_base_new/preds_{n_out}_t{trial}.npy', pca_preds)


            # my_wpca = wpca.WPCA(n_components = 10)
            # my_wpca.fit(Xcenter)
            # Wwpca = my_wpca.components_.T
            # pca_errs = []
            # for i in range(len(X)):   
            #     x = Xcenter[[i],:]
            #     pca_errs.append(np.linalg.norm(x @ Wwpca @ Wwpca.T - x))

            # wpca_preds = np.array(pca_errs)
            # wpca_preds = wpca_preds/np.max(wpca_preds)

            # np.save(f'./results/wpca_base_new/preds_{n_out}_t{trial}.npy', wpca_preds)


            # my_wpca = wpca.EMPCA(n_components = 10)
            # my_wpca.fit(Xcenter)
            # Wwpca = my_wpca.components_.T
            # pca_errs = []
            # for i in range(len(X)):   
            #     x = Xcenter[[i],:]
            #     pca_errs.append(np.linalg.norm(x @ Wwpca @ Wwpca.T - x))

            # wpca_preds = np.array(pca_errs)
            # wpca_preds = wpca_preds/np.max(wpca_preds)

            # np.save(f'./results/empca/preds_{n_out}_t{trial}.npy', wpca_preds)

            # # find basis vectors
            # my_wpca = wpca.PCA(n_components = 10)
            # my_wpca.fit(Xcenter)
            # Wwpca = my_wpca.components_.T

            # pca_errs = []
            # for i in range(len(X)): # for each data point
            #     x = Xcenter[[i],:] # x is point i as a row vector
            #     # x @ Wwpca @ Wwpca.T project row vector onto PCA basis
            #     # look how far this projection is from x
            #     pca_errs.append(np.linalg.norm(x @ Wwpca @ Wwpca.T - x))

            # wpca_preds = np.array(pca_errs)
            # wpca_preds = wpca_preds/np.max(wpca_preds) #normalize by maximum value

            # np.save(f'./results/pca_theirs/preds_{n_out}_t{trial}.npy', wpca_preds)


            # find basis vectors
            Wrpca,_,_ = l1pca_sbfk(Xcenter.T, 10, 200, True)

            pca_errs = []
            for i in range(len(X)): # for each data point
                x = Xcenter[[i],:] # x is point i as a row vector
                # x @ Wwpca @ Wwpca.T project row vector onto PCA basis
                # look how far this projection is from x
                pca_errs.append(np.linalg.norm(x @ Wrpca @ Wrpca.T - x))

            rpca_preds = np.array(pca_errs)
            rpca_preds = rpca_preds/np.max(rpca_preds) #normalize by maximum value

            np.save(f'./results/l1_rpca/preds_{n_out}_t{trial}.npy', rpca_preds)