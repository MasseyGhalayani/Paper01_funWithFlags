'''
Tools for dimensionality reduction using the flag manifolds.

by Nathan Mankovich
'''

import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pymanopt
import scipy

import warnings


def assign_wij(B: np.array, X: np.array, j: int, Ii: np.array, pca_type: str) -> float:
    '''
    Assign Wij for data point j and flag type encoded in Ii
    '''

    # select the jth point
    Xj = X[:,[j]]

    # choose one of these pca types
    if pca_type in ['wpca']:
        wij = np.linalg.norm(Xj - B @ Ii @ B.T @ Xj)
    elif pca_type in ['rpca', 'dpcp']:
        wij = np.linalg.norm(B @ Ii @ B.T @ Xj)
    else:
        print(f'pca_type {pca_type} not recognized')


    #fixed param to avoid dividing by zero
    eps = 10e-8
    wij = np.max([wij, eps])


    return 1/wij

def construct_eyes(flag_type: list) -> list:
    '''
    Construct I_j for flag type
    '''

    # the dimension of the largest subspace in the flag
    n_k = flag_type[-1]

    # assign ''sparse'' identity matrices using flag type
    id_mats = []
    for i in range(len(flag_type)):
        f_type = flag_type[i]
        if i-1 < 0:
            f_type_prev = 0
        else:
            f_type_prev = flag_type[i-1]

        #make identity matrix
        id_mat = np.zeros((n_k,n_k))
        id_mat[np.arange(f_type_prev,f_type,1),np.arange(f_type_prev,f_type,1)] = 1
        id_mats.append(id_mat)

    return id_mats

def solve_stiefel(A: list, B: np.array, pca_type: str, Is: list) -> np.array:
    '''
    Solve flag optimization by optimizing over Stiefel manifold
    '''

    n, n_k = B.shape

    #initialize a stiefel manifold
    St = pymanopt.manifolds.stiefel.Stiefel(n,n_k)


    # generate projection matrices for dpcp
    # note, we dont need these for rpca or wpca
    if pca_type  == 'dpcp':
        [X, Ws] = A
        k = len(Is)
        Xs = []
        for W in Ws:
            Xs.append(X @ np.diag(W) @ X.T)


    # cost function for rpca and wpca
    if pca_type in ['rpca', 'wpca']:
        @pymanopt.function.autograd(St) 
        def cost(point):
            f = -np.trace( A @ point)
            return f

    # cost function for dpcp
    elif pca_type == 'dpcp':
        @pymanopt.function.autograd(St) 
        def cost(point):
            f = 0
            for i in range(k):
                f += np.trace(Xs[i] @ point @ Is[i] @ point.T)
            return f
        
    else:
        print(f'pca_type {pca_type} not recognized')    


    #initialize cost function
    problem = pymanopt.Problem(St, cost)

    # initialize CGD optimizer over stiefel manifold(600 iterations)
    optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient(verbosity = 0, max_iterations= 600)
    # optimizer = pymanopt.optimizers.steepest_descent.SteepestDescent(verbosity = 2, min_step_size = 1e-4) 
    # optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient(verbosity = 2, max_iterations= 2000, orth_value = 1e-4)
    #, min_step_size = 1e-3) #, line_searcher = pymanopt.optimizers.line_search.BackTrackingLineSearcher())#, beta_rule = 'HagerZhang')

    # solve optimization over stiefel manifold
    result = optimizer.run(problem, initial_point = B)
    
    # store solution
    res_pt = result.point

    return res_pt

def objective_val(X: np.array, B: np.array, id_mats: list, pca_type: str, squared: bool = False) -> float:
    '''
    Compute objective function values

        for wpca and dpcp-max:
            \sum_{i=1}^k \sum_{j=1}^p \|U_i U_i^T x_j\|_2
        for rpca and dpdp:
            \sum_{i=1}^k \sum_{j=1}^p \|x_j - U_i U_i^T x_j\|_2
    '''

    f = 0 

    for id_mat in id_mats:
        # go through each subspace in the flag

        for j in range(X.shape[1]):
            #take jth data point
            d = X[:,[j]]

            if pca_type in ['wpca']:
                cos_sq = np.linalg.norm(d - B @ id_mat @ B.T @ d)

            elif pca_type in ['rpca','dpcp']:
                cos_sq = np.linalg.norm( B @ id_mat @ B.T @ d)

            else:
                print(f'pca_type {pca_type} not recognized')

            # if not robust
            if squared:
                cos_sq = cos_sq**2
                
            # add em up over all datapoints and subspaces in flag
            f += cos_sq

    return f

def initialize(data: np.array, pca_type: str, init: str, rand_seed: int, n_k: int, n: int):
    '''
    Make an initial principal direcitons guess options are chosen by the init input

    init: str
        'svd' take the first (for rpca or wpca) or last (for dpcp) n_k left singular vectors of data matrix
        'noisy_svd' same as svd, but with added noise
        'eye' the identity matrix
        'pca' regular pca of data
        'rand' first n_k vectors of Q from QR decomposition of a random matrix (normal distribution, mean 0, std 1)
        you can also just pass an initial point in as init (I know it's bad coding... :/)
    '''


    if init == 'svd':
        U,_,_ = scipy.linalg.svd(data)
        if pca_type in ['dpcp']:
            B0 = U[:,-n_k:]
            np.random.default_rng().shuffle(B0, axis = 1)
        elif pca_type in ['rpca', 'wpca']:
            B0 = U[:,:n_k]
        else:
            print(f'pca_type {pca_type} not recognized')

    elif init == 'noisy_svd':
        U,_,_ = scipy.linalg.svd(data + np.random.normal(0,.01,size = data.shape))

        if pca_type in ['dpcp']:
            B0 = U[:,-n_k:]
        elif pca_type in ['rpca', 'wpca']:
            B0 = U[:,:n_k]
        else:
            print(f'pca_type {pca_type} not recognized')

    elif init == 'eye':
        B0 = np.eye(n)[:,:n_k]

    elif init == 'pca':
        my_pca = PCA(n_components=n_k)
        B0 = my_pca.fit(data.T).components_.T

    elif init == 'rand':
        np.random.seed(rand_seed)
        Q,_ = np.linalg.qr(np.random.normal(0,1,size = (n,n_k)))
        B0 = Q[:,:n_k]

    elif type(init) != str:
        B0 = init
    
    return B0

def flag_robust_pca(data: np.array, flag_type: list, pca_type: str, max_iters: int = 50, 
                  init: str = 'svd', return_all: bool = False, verbose: bool = False, 
                  rand_seed: int = 123):
    '''
    Flags of principal directions for fRPCA, fWPCA, and fDPCP

    Inputs
        data:  p-samples of n-random variables
        flag_type: list of nested subspace dimensions (smallest to largest)
        pca_type: type of pca. options are 'rpca', 'wpca', or 'dpcp'
        max_iters: max number of iterations for irls scheme (default 50)
        init: initialization strategy (default is svd)
        return_all: return principal directions (default False) or 
            also return objective function values for each iteration and 
            chordal flag distance between consecutive iterates (if True)
        verbose: if True print status at each iteration (default False)
        rand_seed: seed for initial guess at PDs (default 123)

    Outputs
        return_all = True: 
            B0 (np.array) matrix of principal directions (as columns)
        return_all = False: 
            tuple of:
                B0 (np.array) matrix of principal directions (as columns)
                err (list) objective function values for each iteration 
                cauchs (list) chordal flag distance between consecutive iterates
    '''

    # p samples of n random variables
    n, p = data.shape
 
    # biggest subspace dimension in the flag
    n_k = flag_type[-1]

    # number of nested subspaces 
    k = len(flag_type)
    
    # construct identity matrices
    id_mats = construct_eyes(flag_type)

    # initial guess for principal directions
    B0 = initialize(data, pca_type, init, rand_seed, n_k, n)

    # initial objective function value
    err = []
    err.append(objective_val(data, B0, id_mats, pca_type))

    cauchs = []

    #some variables to enter the loop
    diff = 1
    cauch = 1

    #iteration number
    it_num = 0

    while it_num < max_iters and cauch > 10e-10 and diff > 1e-10:
        # run until you've reached max_iters or cauch or diff has gotten really small

        # fill an array with all the weights
        wijs = np.zeros((k,p))
        for j in range(p):
            for i in range(k):
                wijs[i,j] = assign_wij(B0, data, j, id_mats[i], pca_type = pca_type) 


        # compute A if rpca or wpca
        if pca_type  in ['rpca', 'wpca']:
            A = compute_A(wijs, data, B0, id_mats)  
        elif pca_type  == 'dpcp': 
            A = [data, wijs]
        else:
            print(f'{pca_type} not recognized')

        # optimize over the stiefel manifold
        B1 = solve_stiefel(A, B0, pca_type, id_mats)
        
        # record distance between previous principal directions and new ones
        cauch = chordal_dist(B0, B1, flag_type)
        cauchs.append(cauch)

        # compute objective function value
        err.append(objective_val(data, B1, id_mats, pca_type))
        
        # compute difference between objective function values
        if it_num > 1:
            diff = np.abs(err[-1]-err[-2])

        B0 = B1.copy()


        # print status
        if verbose:
            print('--------------------')
            
            print(f'iteration {it_num}')

            print(f'objective {err[-1]}')

            print(f'flag distance between B0 and B1 {cauchs[-1]}')
            print()


        it_num += 1
    if return_all:
        return B0, err, cauchs

    else:
        return B0

def chordal_dist(X: np.array, Y: np.array, flag_type: list)-> float:

    '''
    Chordal distance on flag manifold between X and Y (of flag type flag_type)

    note: flag_type is subspace dimensions of flag in increasing order
    '''

    c_dist = 0
    for i in range(len(flag_type)): #loop through each subspace in flags

        # get subspace indices
        f_type = flag_type[i]
        if i < 1:
            f_type_prev = 0
        else:
            f_type_prev = flag_type[i-1]
        k_i = f_type-f_type_prev
        
        # store the piece of the flag for X and Y
        dimX = Y[:,f_type_prev:f_type]
        dimY = X[:,f_type_prev:f_type]

        #squared chordal distance between dimX and dimY
        c_dist += k_i - np.trace(dimY.T @ dimX @ dimX.T @ dimY)

    #avoid taking square roots of negative numbers (for numerical errors)
    if c_dist < 0:
        c_dist = 0
        # print('warning: distance is close to 0')

    # take the square root to get chordal distance between flag X and flag Y
    c_dist = np.sqrt(c_dist)

    return c_dist

def compute_A(wijs: list, X: np.array, B: np.array, Is: list) -> np.array:
    '''
    Compute A([[B]]) with data X, weights wijs, and list of identity matrices Is
    '''

    n,n_k = B.shape

    A = np.zeros((n_k,n))
    for i in range(len(Is)):
        Mj =  X @ np.diag(wijs[i,:]) @ X.T
        Aj = Is[i] @ B.T @ Mj
        A += Aj

    return A

def weighted_flag_pca(X: np.array, Ws: list, n: int, pca_type: str, initial_guess: np.array, fl_type: list)-> np.array:
    '''
    compute weighted fPCA usign weights in Ws

    '''
    n_k = fl_type[-1]
    k = len(fl_type)

    Is = construct_eyes(fl_type)

    Xs = []
    for W in Ws:
        Xs.append(X @ np.diag(W) @ X.T)

    St = pymanopt.manifolds.stiefel.Stiefel(n,n_k)

    if pca_type in ['rpca']:

        @pymanopt.function.autograd(St) 
        def cost(point):
            f = 0
            for i in range(k):
                f -= np.trace(Xs[i] @ point @ Is[i] @ point.T)
            return f

    elif pca_type in ['wpca','dpcp']:

        @pymanopt.function.autograd(St) 
        def cost(point):
            f = 0
            for i in range(k):
                f = np.trace(Xs[i] @ point @ Is[i] @ point.T)
            return f

    else:
        print(f'pca_type {pca_type} not recognized')


    problem = pymanopt.Problem(St, cost)


    optimizer = pymanopt.optimizers.trust_regions.ConjugateGradient(verbosity = 1)

    result = optimizer.run(problem, initial_point = initial_guess) #this throws a warning.

    res_pt = result.point

    return res_pt

def flag_robust_tpca(manifold, data: list,fl_type: list, pca_type: str, verbose: bool = False, return_all: bool = True, 
          max_iters: int = 200, init: str =  'svd', eps: float = 1e-9, return_ts: bool = False, rand_seed: int = 123,
          median = []):
    
    if len(median) == 0:
        median = karcher_median(manifold, data, eps = eps)

    t_data = []
    for i in range(len(data)):
        t = manifold.log(median, data[i]).flatten()
        t_data.append(t)

    stacked_data = np.vstack(t_data).T

    t_data = stacked_data

    if return_ts:
        return flag_robust_pca(t_data, fl_type, pca_type, verbose = verbose, return_all = return_all, max_iters = max_iters, init = init, rand_seed=rand_seed), t_data
    else:
        return flag_robust_pca(t_data, fl_type, pca_type, verbose = verbose, return_all = return_all, max_iters = max_iters, init = init, rand_seed=rand_seed)
 
def karcher_median(manifold, data: list, step_size: float = .05, eps: float = 1e-9, seed: int = 42):
    '''
    Karcher mean on flag manifold
    '''
    np.random.seed(seed)
    mean_old  = manifold.random_point()

    err = 1
    while err > eps:

        t_data = []
        wts = []
        for i in range(len(data)):
            dist = manifold.dist(mean_old, data[i])
            wt = 1/np.max([dist, eps])
            wts.append(wt)
            t_data.append(wt*manifold.log(mean_old, data[i]))

        t_mean = np.sum(t_data, axis = 0)/np.sum(wts)

        mean_new = manifold.exp(mean_old, step_size*t_mean)

        err = np.linalg.norm(mean_new - mean_old)**2

        mean_old = mean_new.copy()
    
    return mean_old

def karcher_mean(manifold, data: list, weights: list = [], step_size: float = .05, eps: float = 1e-9, seed: int = 42):
    np.random.seed(seed)
    mean_old  = manifold.random_point()
    err = 1

    if len(weights) == 0:
        weights = [1]*len(data)

    while err > eps:
        t_data = []
        for i in range(len(data)):
            t_data.append(weights[i]* manifold.log(mean_old, data[i]))
        t_mean = np.mean(t_data, axis = 0)
        mean_new = manifold.exp(mean_old, step_size*t_mean)

        err = np.linalg.norm(mean_new - mean_old)**2

        mean_old = mean_new

    return mean_new

def mean_center(data):
    if type(data) == list:
        data = np.vstack(data).T
    data = data.T - np.mean(data, axis = 1)
    return data.T
