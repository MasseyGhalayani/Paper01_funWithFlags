#data and reps from here:
#https://github.com/henanjun/LCMR/blob/1/LCMR/LCMR_demo.m

#demo for LCMR to be appeared TGRS if you use code, please kind cite the paper, "A New Spatial-Spectral Feature Extraction Method for Hyperspectral Images Using Local Covariance Matrix Representation" Leyuan Fang et al. to be appeared TGRS

import numpy as np

from hand_utils import *
 
import sys
sys.path.append('../scripts')
import flag_dimensionality_reduction as fdr
from utils import *


from matplotlib import pyplot as plt
from geomstats.geometry.pre_shape import PreShapeSpace

import pandas as pd



class GeomstatsShapeSpace():
    """A simple adapter class which uses pymanopt language for geomstats shape space
    """

    def __init__(self, n, k):
        self._man = PreShapeSpace(m_ambient = n, k_landmarks = k)
        self._man.equip_with_group_action("rotations")
        self._man.equip_with_quotient_structure()

    def dist(self, point_a, point_b):
        return self._man.metric.dist(point_a, point_b)

    def exp(self, base_point, tangent_vector):
        return self._man.metric.exp(tangent_vector, base_point=base_point)

    def log(self, base_point, ambient_point):
        return self._man.to_tangent(ambient_point, base_point=base_point)
    
    def random_point(self):
        return self._man.random_point()

def generate_ellipse_outlier1(seed: int):

    np.random.seed(seed)
    # Define ellipse parameters
    center_x = .1*np.random.normal(0,1)  # X-coordinate of the center #was .1
    center_y = .1*np.random.normal(0,1)  # Y-coordinate of the center #was .1
    major_axis = .4+ np.random.normal(0,.1)#np.random.normal(0,.3) # np.random.normal(0,1)  # Length of the major axis
    minor_axis = .4+ np.random.normal(0,.1)#np.random.normal(0,.3)  #np.random.normal(0,1)   # Length of the minor axis


    # Generate data points on the ellipse
    theta = np.linspace(.1 + np.random.rand(),2*np.pi, 56)
    x = center_x + major_axis * np.cos(theta)
    y = center_y + minor_axis * np.sin(theta)
    
    pt = np.vstack([x,y]).T

    pt = pt - np.mean(pt, axis = 0)
    pt = pt/np.linalg.norm(pt)
    return pt

if __name__ == '__main__':

    dr_seed = 12345

    manifold = GeomstatsShapeSpace(2,56)

    results = pd.DataFrame(columns = ['Num. Outliers','Algorithm', 'AUC'])
    fl_type = [1,2]

    #33 min to run

    for trial in range(5):

        for n_outs in [10,20,30,40,50,60,70,80,90]:
            

            # load data
            filepath = './all/shapes'
            segmentationlist = readSegmentations(filepath,getxy)[0]
            hands0 = np.array(segmentationlist).T
            hands = procrustes_hand(hands0)

            hands1 = []
            for h in hands:
                k_shape = unmake_1d(h)
                k_shape = k_shape - np.mean(k_shape, axis = 0)
                k_shape = k_shape/np.linalg.norm(k_shape)
                hands1.append(k_shape)


            x = hands1[0]

            ys = [hands1[7],hands1[9]] #works with 7

            np.random.seed(42)

            shapes = []
            for i in np.linspace(-1,1,40):
                idx = np.random.choice([0,1])
                v = manifold.log(x,ys[idx])
                inlier = manifold.exp(x, i*v)
                shapes.append(inlier)

            np.random.seed(trial)
            for i in range(n_outs):
                # x= unmake_1d(hands[np.random.choice(np.arange(11,40,8))])
                x= unmake_1d(hands[35])
                x = x - np.mean(x, axis = 0)
                x = x/np.linalg.norm(x)
                y = manifold.random_point()
                v = manifold.log(x,y)
                outlier = manifold.exp(x, .04*np.random.rand()*v) #.05
                shapes.append(outlier)
            # x = generate_ellipse_outlier1(trial)
            # for i in range(n_outs):
            #     # x= unmake_1d(hands[np.random.choice(np.arange(11,40,8))])
            #     # x= unmake_1d(hands[35])
            #     x = x - np.mean(x, axis = 0)
            #     x = x/np.linalg.norm(x)
            #     y = manifold.random_point()
            #     v = manifold.log(x,y)
            #     outlier = manifold.exp(x, .04*np.random.rand()*v) #.05
            #     shapes.append(outlier)

            labels = [0]*40 + [1]*n_outs


            #compute median
            mean_seed = 21
            median = fdr.karcher_median(manifold, shapes, seed = mean_seed)

            mean = fdr.karcher_mean(manifold, shapes, seed = mean_seed)


            [W_w1ours,_,_], ts     = fdr.flag_robust_tpca( manifold, shapes, fl_type = fl_type, pca_type = 'wpca', return_ts = True, 
                                                median = median, init = 'rand', rand_seed = dr_seed, verbose = False )
            [W_r1ours,_,_], ts     = fdr.flag_robust_tpca( manifold, shapes, fl_type = fl_type, pca_type = 'rpca', return_ts = True,
                                                median = median, init = 'rand', rand_seed = dr_seed, verbose = False )
            [W_d1ours,_,_], ts     = fdr.flag_robust_tpca( manifold, shapes, fl_type = fl_type, pca_type = 'dpcp', return_ts = True,
                                                median = median, init = 'rand', rand_seed = dr_seed, verbose = False )
            
            [W_w2ours,_,_], ts     = fdr.flag_robust_tpca( manifold, shapes, fl_type = [fl_type[-1]], pca_type = 'wpca', return_ts = True, 
                                                median = median, init = 'rand', rand_seed = dr_seed, verbose = False )
            [W_r2ours,_,_], ts     = fdr.flag_robust_tpca( manifold, shapes, fl_type = [fl_type[-1]], pca_type = 'rpca', return_ts = True,
                                                median = median, init = 'rand', rand_seed = dr_seed, verbose = False )
            [W_d2ours,_,_], ts     = fdr.flag_robust_tpca( manifold, shapes, fl_type = [fl_type[-1]], pca_type = 'dpcp', return_ts = True,
                                                median = median, init = 'rand', rand_seed = dr_seed, verbose = False )

            #PCA
            W_pca, _, ts_pca, mean =    PGA0( manifold, shapes, [fl_type[-1]], return_ts = True, eps = 1e-9 )

            auc_val, _, _ = run_roc( shapes, W_w1ours, labels, 'wpca', manifold, ts,     median, do_plots = False )
            row = pd.DataFrame(data = [[n_outs, 'L1-WPCA', auc_val]],
                            columns = results.columns)
            results = pd.concat([results, row])
            
            auc_val, _, _= run_roc( shapes, W_r1ours, labels, 'rpca', manifold, ts,     median, do_plots = False )
            row = pd.DataFrame(data = [[n_outs, 'L1-RPCA', auc_val]],
                            columns = results.columns)
            results = pd.concat([results, row])
                
            auc_val, _, _ = run_roc( shapes, W_d1ours, labels, 'dpcp', manifold, ts,     median, do_plots = False )
            row = pd.DataFrame(data = [[n_outs, 'L1-DPCP', auc_val]],
                            columns = results.columns)
            results = pd.concat([results, row])

            auc_val, _, _ = run_roc( shapes, W_w2ours, labels, 'wpca', manifold, ts,     median, do_plots = False )
            row = pd.DataFrame(data = [[n_outs, 'L2-WPCA', auc_val]],
                            columns = results.columns)
            results = pd.concat([results, row])
            
            auc_val, _, _= run_roc( shapes, W_r2ours, labels, 'rpca', manifold, ts,     median, do_plots = False )
            row = pd.DataFrame(data = [[n_outs, 'L2-RPCA', auc_val]],
                            columns = results.columns)
            results = pd.concat([results, row])
                
            auc_val, _, _= run_roc( shapes, W_d2ours, labels, 'dpcp', manifold, ts,     median, do_plots = False )
            row = pd.DataFrame(data = [[n_outs, 'L2-DPCP', auc_val]],
                            columns = results.columns)
            results = pd.concat([results, row])
            
            auc_val, _, _ = run_roc( shapes, W_pca,    labels, 'wpca', manifold, ts_pca, mean,   do_plots = False )
            row = pd.DataFrame(data = [[n_outs, 'PCA', auc_val]],
                            columns = results.columns)
            results = pd.concat([results, row])

            print(results)
            
            results.to_csv(f'../Results/hand_outlier_res_debug{trial}.csv')


    results = pd.read_csv('../Results/hand_outlier_res_debug4.csv', index_col = 0)

    max_res = pd.DataFrame(columns = results.columns)

    for outl in np.unique(results['Num. Outliers']):
        for alg in np.unique(results['Algorithm']):
            idx = (results['Num. Outliers'] == outl) & (results['Algorithm'] == alg)
            row = pd.DataFrame(columns = results.columns, data = [[outl, alg, np.mean(results[idx]['AUC'])]])
            max_res = pd.concat([max_res, row])

    results = max_res.copy()




    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize = (8,3))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'indigo', 'maroon', 'teal', 'orange', 'purple', 'brown']
    linestyles = ['--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1, 1, 1))]
    ii = 0

    for alg, lbl in zip(['PCA', 'L2-WPCA', 'L1-WPCA', 'L2-RPCA',  'L1-RPCA',  'L2-DPCP', 'L1-DPCP'],['TPCA',  'fWTPCA(k)', 'fWTPCA(1,...,k)', 'fRTPCA(k)', 'fRTPCA(1,...,k)', 'fTDPCP(k)', 'fTDPCP(1,...,k)']):

        y = results[results['Algorithm'] == f'{alg}']['AUC']
        x = results[results['Algorithm'] == f'{alg}']['Num. Outliers']
        plt.plot(x/(40+x),y,label = lbl, c = colors[ii], linestyle = linestyles[ii], alpha = .5, linewidth = 5)

        ii+=1

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 14)
    plt.xlabel('Prop. Outliers')
    plt.ylabel('AUC')



    plt.tight_layout()
    plt.savefig('../Results/hand_outlier_res_debug.pdf', bbox_inches = 'tight')