import numpy as np

import sys
sys.path.append('../')
from scripts import flag_dimensionality_reduction as fdr
from scripts.utils import *


import time


from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


def ftype_roc(data, W, labels, alg = 'wpca'):

    if alg in ['wpca','rpca']:
        preds = reconst_class(data, W)
    elif alg == 'dpcp':
        preds = min_var_class(data, W)


    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(alg)
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc


if __name__ == '__main__':
    np.random.seed(42)
    X = .1*np.random.rand(10,300)
    X[[0,1],:150] = np.random.rand(2,150)
    X[[2,3,4],150:200] = np.random.rand(3,50)
    X[[5,6],200:300] = np.random.rand(2,100)


    column_means = np.mean(X.T, axis=0)
    Xcenter = X.T - column_means
    Xcenter = Xcenter.T

    unitX = np.vstack([x/np.linalg.norm(x) for x in X.T]).T


    fl_type = [2,5,9]

    pca_type = 'wpca'

    start = time.time()
    Wrpca, errs, cauchs = fdr.flag_robust_pca(Xcenter, fl_type, pca_type, verbose = False, return_all = True, max_iters = 200, init = 'rand')
    print(f'elapsed time: {time.time() - start}')


    ftype_roc(Xcenter, Wrpca[:,:2], np.array([0]*150+[1]*150))

    ftype_roc(Xcenter, Wrpca[:,2:5], np.array([1]*150+[0]*50+[1]*100))

    ftype_roc(Xcenter, Wrpca[:,5:7], np.array([1]*200+[0]*100))



    fl_type = [9]

    pca_type = 'wpca'

    start = time.time()
    Wrpca, errs, cauchs = fdr.flag_robust_pca(Xcenter, fl_type, pca_type, verbose = False, return_all = True, max_iters = 200, init = 'rand')
    print(f'elapsed time: {time.time() - start}')


    ftype_roc(Xcenter, Wrpca[:,:2], np.array([0]*150+[1]*150))

    ftype_roc(Xcenter, Wrpca[:,2:5], np.array([1]*150+[0]*50+[1]*100))

    ftype_roc(Xcenter, Wrpca[:,5:7], np.array([1]*200+[0]*100))






