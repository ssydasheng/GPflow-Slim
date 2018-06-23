
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import gpflowSlim as gpf
import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def standardize(data_train, *args):
    """
    Standardize a dataset to have zero mean and unit standard deviation.
    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.
    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    output = [data_train_standardized]
    for d in args:
        dd = (d - mean) / std
        output.append(dd)
    output.append(mean)
    output.append(std)
    return output


def load_boston_housing(seed=1231):
    x, y = load_boston(True)
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.1, random_state=seed)

    x_t, x_v, _, _ = standardize(x_t, x_v)
    y_t, y_v, _, train_std = standardize(y_t, y_v)
    return x_t, y_t, x_v, y_v, train_std

# inputs and targets from Boston
x_train, y_train, x_test, y_test, std_y_train = load_boston_housing()
x_train, y_train = x_train.astype(gpf.settings.float_type), y_train.astype(gpf.settings.float_type)
x_test, y_test = x_test.astype(gpf.settings.float_type), y_test.astype(gpf.settings.float_type)

# TODO: test when ARD=FALSE
k = gpf.kernels.RBF(13, ARD=True)
m = gpf.models.GPR(x_train, np.expand_dims(y_train, 1), kern=k)
# m.likelihood.variance = 0.03

objective = m.objective
optimizer = tf.train.AdamOptimizer(1e-3)
infer = optimizer.minimize(objective)

pred_mu, pred_cov = m.predict_f(x_test)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(20000):
        _, obj = sess.run([infer, objective])
        if iter % 10 == 0:
            print('Iter {}: Loss = {}'.format(iter, obj))

            # evaluate
            mu, cov = sess.run(
                [pred_mu, pred_cov]
            )
            mu, cov = mu.squeeze(), cov.squeeze()
            rmse = np.mean((mu - y_test) ** 2) ** .5 * std_y_train

            log_likelihood = np.mean(np.log(stats.norm.pdf(
                y_test,
                loc=mu,
                scale=cov ** 0.5))) - np.log(std_y_train)
            print('test rmse = {}'.format(rmse))
            print('tset ll = {}'.format(log_likelihood))
