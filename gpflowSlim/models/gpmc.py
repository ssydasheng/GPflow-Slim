# Copyright 2018 Shengyang Sun
# Copyright 2016 James Hensman, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import tensorflow as tf

from .. import settings
from ..params import Parameter
from ..priors import Gaussian
from ..conditionals import conditional

from .model import GPModel


class GPMC(GPModel):
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None,
                 num_latent=None,
                 **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        This is a vanilla implementation of a GP with a non-Gaussian
        likelihood. The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            f = Lv + m(x)

        with

            L L^T = K

        """
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]

        with tf.variable_scope(self.name):
            self._V = Parameter(np.zeros((self.num_data, self.num_latent)), name='V')
        self._V.prior = Gaussian(0., 1.)

        self._parameters = self._parameters + [self._V]

    @property
    def V(self):
        return self._V.value

    def _build_likelihood(self):
        """
        Construct a tf function to compute the likelihood of a general GP
        model.

            \log p(Y, V | theta).

        """
        K = self.kern.K(self.X)
        L = tf.cholesky(
            K + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * settings.numerics.jitter_level)
        F = tf.matmul(L, self.V) + self.mean_function(self.X)

        return tf.reduce_sum(self.likelihood.logp(F, self.Y))

    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (F=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at X.

        """
        mu, var = conditional(Xnew, self.X, self.kern, self.V,
                              full_cov=full_cov,
                              q_sqrt=None, white=True)
        return mu + self.mean_function(Xnew), var
