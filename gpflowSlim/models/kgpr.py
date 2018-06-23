from __future__ import absolute_import
import tensorflow as tf
import numpy as np

from .. import likelihoods
from .. import settings

from ..decors import name_scope
from ..mean_functions import Zero
from .model import Model
from ..conjugate_gradient import vec, cgsolver, dot


class KGPR(Model):
    def __init__(self, X1, X2, Y, kern1, kern2, mask, mean_function=None, obs_var=0.1, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        if 'name' in kwargs:
            with tf.variable_scope(kwargs['name']):
                self.likelihood = likelihoods.Gaussian(var=obs_var)
                self.kern1 = kern1
                self.kern2 = kern2
                self.mean_function = mean_function or Zero()
        else:
            self.likelihood = likelihoods.Gaussian(var=obs_var)
            self.kern1 = kern1
            self.kern2 = kern2
            self.mean_function = mean_function or Zero()
        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.N = np.prod(Y.shape)
        self.M = self.N - np.sum(mask)

        self.noise = vec(self.likelihood.variance * np.ones_like(Y) + mask * 1e6)
        Model.__init__(self, **kwargs)
        self._parameters = self.mean_function.parameters + self.kern1.parameters \
                           + self.kern2.parameters + self.likelihood.parameters

    @name_scope('likelihood')
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K1 = self.kern1.K(self.X1)
        K2 = self.kern2.K(self.X2)
        e1, _ = tf.self_adjoint_eig(K1)
        e2, _ = tf.self_adjoint_eig(K2)
        e1 = tf.expand_dims(e1, 0)
        e2 = tf.expand_dims(e2, 1)

        e, _ = tf.nn.top_k(tf.reshape(tf.matmul(e2, e1), [-1]), k=self.M)
        e = e * self.M / self.N
        logdet = tf.reduce_sum(tf.log(e + self.likelihood.variance))

        y = vec(self.Y)
        C = self.noise ** (-0.5)
        Ay = C * cgsolver(K1, K2, C * y, C)


        quadratic = dot(y, Ay)

        return - 0.5 * logdet - 0.5 * quadratic - 0.5 * self.M * np.log(2 * np.pi)


    @name_scope('predict')
    def _build_predict(self, Xnew1, Xnew2):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        K1 = self.kern1.K(self.X1)
        K2 = self.kern2.K(self.X2)
        K1u = self.kern1.cov(self.X1, Xnew1)
        K2u = self.kern2.cov(self.X2, Xnew2)
        m = tf.shape(K1)[0]
        n = tf.shape(K2)[0]

        y = vec(self.Y)
        C = self.noise ** (-0.5)
        Ky = C * cgsolver(K1, K2, C * y, C)

        fmean = tf.matmul(tf.matmul(K1u, tf.transpose(tf.reshape(Ky, [n, m])), transpose_a=True), K2u)
        return fmean

    def predict_f(self, Xnew1, Xnew2):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self._build_predict(Xnew1, Xnew2)
