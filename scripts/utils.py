import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

import sys
sys.path.append('../scripts')
import flag_dimensionality_reduction as fdr


def outlier_gen(manifold, n_in: int, n_out: int, seed = 111111):
    '''
    Generate a dataset of inliers and outliers on a manifold
    '''

    np.random.seed(seed)

    #randomly sample an inlier center
    center = manifold.random_point()
    #randomly sample two tangent vectors from the same center
    t0 = manifold.random_tangent_vector(center)
    t1 = manifold.random_tangent_vector(center)
    ts = [t0, t1]

    #randomly sample outlier center
    center_o = manifold.random_point()

    
    data_all = []
    #sample inlier data
    for _ in range(n_in):
        #sample an inlier tangent vector
        t = ts[np.random.choice([0,1])]
        #go a random amount away from the center along an inlier tangent vectors
        Y = manifold.exp(center, np.random.rand()*t)
        data_all.append(Y)

    # sample outlier data
    for _ in range(n_out):
        #sample an outlier tangent vector
        t_o = manifold.random_tangent_vector(center_o)
        #randomly go a small amount away from the outlier center
        Y = manifold.exp(center_o, 0.1*np.random.rand()*t_o)
        data_all.append(Y)

    return data_all

def PGA0(manifold, data: list, fl_type: list, return_ts: bool = False, eps: float = 1e-5):
    '''
    A function for principal geodesic analysis
    '''

    #compute the karcher mean
    mean = fdr.karcher_mean(manifold, data, eps = eps)

    #translate data to flattened tangent vectors at the mean
    t_data = []
    for i in range(len(data)):
        t = manifold.log(mean, data[i]).flatten()
        t_data.append(t)

    #stack tangent vector data
    t_data = np.vstack(t_data).T


    #do pca on the tangent vectors and take the first fl_type[-1] principal directions
    pds = np.linalg.svd(t_data)[0][:,:fl_type[-1]]

    if return_ts:
        return pds, '', t_data, mean
    else:
        return pds, ''

def reconst_class(data: np.array, W: np.array):
    '''
    generate a reconstruction score for a set of predictions
        1 = outlier
        0 = inlier
    '''

    errs = []
    for i in range(len(data.T)):   
        x = data[:,i]
        #compute reconstruction error for x
        errs.append(np.linalg.norm(x @ W @ W.T - x))

    preds = np.array(errs)

    #normalize by the maximum reconstruction error
    preds = preds/np.max(preds)

    return preds

def reconst_class_man(manifold, mu: np.array, data: list, t_data: np.array, W: np.array, return_data: bool = False):

    ''' 
    compute reconstructions for the data. 
        1 = outlier
        0 = inlier
    '''

    nrow, ncol = manifold.random_point().shape

    rec_data = []
    for i in range(len(t_data.T)):  
        x = t_data[:,i]

        #project data (as tangent vectors) onto the span of the principal directions (in W)
        t_pt = np.reshape (x @ W @ W.T, (nrow, ncol) )

        #wrap data back onto manifold using mean and 
        rec_data.append( manifold.exp(mu, t_pt))


    errs = []
    for i in range(len(data)):
        #compute reconstruction error as distance on manifold between reconstructed data and the actual data
        errs.append(manifold.dist(rec_data[i], data[i]))

    preds = np.array(errs)

    #normalize by the maximum reconstruction error
    preds = preds/np.max(preds)
    
    if return_data:
        return errs, rec_data
    else:
        return preds
    
def min_var_class(data: np.array, W: np.array):
    ''' 
    compute the norm of the vector of inner products between the data and the principal directions

        1 = outlier
        0 = inlier
    '''
    errs = []
    for i in range(len(data.T)):   
        x = data[:,i]
        # compute norm of vector of inner products
        errs.append(np.linalg.norm(x @ W))
    preds = np.array(errs)

    #normalize by the maximum entry in preds
    preds = preds/np.max(preds)
    return preds

def min_var_class_man(t_data: np.array, W: np.array):
    ''' 
    compute the distance between the mean and the data on the manifold
        1 = outlier
        0 = inlier
    '''
    errs = []
    for i in range(len(t_data.T)):  
        x = t_data[:,i]
        errs.append(np.linalg.norm(x @ W )) 

    preds = np.array(errs)
    preds = preds/np.max(preds)

    return preds

def run_roc_tangent(data, W, labels, pca_type = 'wpca', tangent = False, manifold = None, t_data = None, mu = None):

    if not tangent:
        if pca_type != 'dpcp':
            preds = reconst_class(data, W)
        else:
            preds = min_var_class(data, W)
    else:
        if pca_type != 'dpcp':
            preds = reconst_class_man(manifold, mu, data, t_data, W)
        else:
            preds = min_var_class_man(t_data, W)


    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    return roc_auc

def run_roc(data, W, labels, pca_type = 'wpca', manifold = None, t_data = None, mu = None, do_plots = True):

    if pca_type != 'dpcp':
        preds = reconst_class_man(manifold, mu, data, t_data, W)
    else:
        preds = min_var_class_man(t_data, W) # is (1- bla) for paper

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    if do_plots:
        # # Plot ROC curve
        plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title(alg)
        plt.legend(loc="lower right")
        plt.show()

    return roc_auc, preds, optimal_threshold

def generate_outlier(seed):
    np.random.seed(seed)

    # Generate data points on the ellipse
    x = np.random.normal(0,10, 56)
    y = np.random.normal(0,10, 56) 

    
    pt = np.vstack([x,y]).T

    pt = pt - np.mean(pt, axis = 0)
    pt = pt/np.linalg.norm(pt)
    return pt

def generate_ellipse_outlier(seed: int):
    np.random.seed(seed)
    # Define ellipse parameters
    center_x = .1*np.random.normal(0,1)  # X-coordinate of the center #was .1
    center_y = .1*np.random.normal(0,1)  # Y-coordinate of the center #was .1
    major_axis = .4+ np.random.normal(0,.5)#np.random.normal(0,.3) # np.random.normal(0,1)  # Length of the major axis
    minor_axis = .4+ np.random.normal(0,.5)#np.random.normal(0,.3)  #np.random.normal(0,1)   # Length of the minor axis


    # Generate data points on the ellipse
    theta = np.linspace(-0.2, 1.8*np.pi, 56)  # Create 100 equally spaced points around the ellipse
    x = center_x + major_axis * np.cos(theta)
    y = center_y + minor_axis * np.sin(theta)
    
    pt = np.vstack([x,y]).T

    pt = pt - np.mean(pt, axis = 0)
    pt = pt/np.linalg.norm(pt)
    return pt

def generate_hairball_outlier(seed: int):
    np.random.seed(seed)

    # Generate data points on the ellipse
    x = np.random.normal(0,10, 56)
    y = np.random.normal(0,10, 56) 

    
    pt = np.vstack([x,y]).T

    pt = pt - np.mean(pt, axis = 0)
    pt = pt/np.linalg.norm(pt)
    return pt