# data and reps from here:
# https://github.com/henanjun/LCMR/blob/1/LCMR/LCMR_demo.m

# demo for LCMR to be appeared TGRS if you use code, please kind cite the paper, "A New Spatial-Spectral Feature Extraction Method for Hyperspectral Images Using Local Covariance Matrix Representation" Leyuan Fang et al. to be appeared TGRS

import sys

sys.path.append('./Hands')
from hand_utils import *

sys.path.append('./scripts')
import flag_dimensionality_reduction as fdr
from utils import *

from itertools import compress

from matplotlib import pyplot as plt
from geomstats.geometry.pre_shape import PreShapeSpace


class GeomstatsShapeSpace():
    """A simple adapter class which uses pymanopt language for geomstats shape space
    """

    def __init__(self, n, k):
        self._man = PreShapeSpace(m_ambient=n, k_landmarks=k)
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


n_outs = 16

n_ins = 44 - n_outs

trial = 4

# load data
filepath = './Hands/all/shapes'
segmentationlist = readSegmentations(filepath, getxy)[0]
hands0 = np.array(segmentationlist).T
hands = procrustes_hand(hands0)

np.random.seed(trial)
in_ids = np.random.choice(40, n_ins, replace=False)

shapes = []
for i in in_ids:
    h = hands[i]
    k_shape = unmake_1d(h)
    k_shape = k_shape - np.mean(k_shape, axis=0)
    k_shape = k_shape / np.linalg.norm(k_shape)
    shapes.append(k_shape)

for i in range(n_outs):
    outlier = generate_ellipse_outlier(i + 10 * trial)
    shapes.append(outlier)

labels = [0] * n_ins + [1] * n_outs




flat_data = np.vstack([d.flatten() for d in shapes]).T #features (random variables) are rows and points (samples) are columns

W_fpca, errs, cauchs   = fdr.flag_robust_pca(   flat_data,          # data set in R^n
                                                [1,2],              # flag type (nested subspace dimensions in increasing order)
                                                'wpca',             # robust pca variant (wpca, rpca, or dpcp)
                                                max_iters = 200,    # max number of irls itertions
                                                init= 'rand',       # initialization type
                                                verbose = True,     # print progress at each iteration
                                                return_all = True)  # return more than the principal directions?


# plot results
fig, [ax1, ax2] = plt.subplots(1,2, figsize = (8,2))

ax1.plot(errs)
ax1.set_xlabel('Iteration')
ax1.set_title('Objective Value')

ax2.plot(cauchs)
ax2.set_xlabel('Iteration')
ax2.set_title('Distance Between Iterates')