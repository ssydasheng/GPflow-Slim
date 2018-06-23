from __future__ import print_function

import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import gpflowSlim as gpf
from gpflowSlim import ekernels

import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pods

def median_distance_local(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    return np.median(dis_a, 0)


pods.datasets.overide_manual_authorize = True  # dont ask to authorize
np.random.seed(42)
gpf.settings.numerics.quadrature = 'error'  # throw error if quadrature is us

data = pods.datasets.oil_100()
Y = data['X'].astype(gpf.settings.float_type)

Q = 5
M = 20  # number of inducing pts
N = Y.shape[0]
X_mean = gpf.models.PCA_reduce(Y, Q) # Initialise via PCA
Z = np.random.permutation(X_mean.copy())[:M]
k = ekernels.Sum([
    ekernels.RBF(3, lengthscales=median_distance_local(X_mean)[:3],
                 ARD=True, active_dims=slice(0,3), name='RBF1'),
    ekernels.RBF(2, lengthscales=median_distance_local(X_mean)[3:],
                 ARD=True, name='RBF2')
    # ekernels.Linear(2, ARD=False, active_dims=slice(3,5), name='Linear')
])
m = gpf.models.BayesianGPLVM(
    X_mean=X_mean, X_var=0.1*np.ones((N, Q)), Y=Y, kern=k, M=M, Z=Z)

obj = m.objective
opt = tf.train.AdamOptimizer(1e-3)
infer = opt.minimize(obj)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(3000):
        _, obj_ = sess.run([infer, obj])
        if iter % 50 == 0:
            print('loss = {}'.format(obj_))

    kern = m.kern.kern_list[0]
    sens, X_mean = sess.run([tf.sqrt(kern.variance) / kern.lengthscales, m.X_mean])

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(sens)) , sens, 0.1, color='y')
    ax.set_title('Sensitivity to latent inputs')

    XPCAplot = gpf.models.PCA_reduce(data['X'], 2)
    f, ax = plt.subplots(1,2, figsize=(10,6))
    labels=data['Y'].argmax(axis=1)
    colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))

    for i, c in zip(np.unique(labels), colors):
        ax[0].scatter(XPCAplot[labels==i,0], XPCAplot[labels==i,1], color=c, label=i)
        ax[0].set_title('PCA')
        ax[1].scatter(X_mean[labels==i,1], X_mean[labels==i,2], color=c, label=i)
        ax[1].set_title('Bayesian GPLVM')
    plt.savefig('bayesian-gplvm.png')
