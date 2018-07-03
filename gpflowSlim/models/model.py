# Copyright 2018 Shengyang Sun
# Copyright 2016 James Hensman, Mark van der Wilk, Valentine Svensson, alexggmatthews, fujiisoup
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

from __future__ import print_function, absolute_import

import abc

import numpy as np
import tensorflow as tf

from .. import settings
from ..mean_functions import Zero
from ..LBFGS import LBFGS
import tensorflow.contrib.eager as tfe


class Model(object):
    def __init__(self, name='model'):
        """
        Name is a string describing this model.
        """
        self._name = name
        self._parameters = []

    @property
    def name(self):
        return self._name

    def compute_log_prior(self):
        """Compute the log prior of the model."""
        return self.prior_tensor

    def compute_log_likelihood(self):
        """Compute the log likelihood of the model."""
        return self.likelihood_tensor

    @property
    def parameters(self):
        return self._parameters

    @property
    def likelihood_tensor(self):
        return self._build_likelihood()

    @property
    def prior_tensor(self):
        priors = []
        for param in self.parameters:
            priors.append(param._build_prior(
                param.unconstrained_tensor, param.constrained_tensor))
        if not priors:
            return tf.constant(0, dtype=settings.float_type)
        return tf.add_n(priors, name='prior')

    @property
    def objective(self):
        func = tf.add(
            self.likelihood_tensor,
            self.prior_tensor,
            name='nonneg_objective')
        return tf.negative(func, name='objective')

    @abc.abstractmethod
    def _build_likelihood(self):
        raise NotImplementedError('') # TODO(@awav): write error message


class GPModel(Model):
    """
    A base class for Gaussian process models, that is, those of the form

    .. math::
       :nowrap:

       \\begin{align}
       \\theta & \sim p(\\theta) \\\\
       f       & \sim \\mathcal{GP}(m(x), k(x, x'; \\theta)) \\\\
       f_i       & = f(x_i) \\\\
       y_i\,|\,f_i     & \sim p(y_i|f_i)
       \\end{align}

    This class mostly adds functionality to compile predictions. To use it,
    inheriting classes must define a build_predict function, which computes
    the means and variances of the latent function. This gets compiled
    similarly to build_likelihood in the Model class.

    These predictions are then pushed through the likelihood to obtain means
    and variances of held out data, self.predict_y.

    The predictions can also be used to compute the (log) density of held-out
    data via self.predict_density.

    For handling another data (Xnew, Ynew), set the new value to self.X and self.Y

    >>> m.X = Xnew
    >>> m.Y = Ynew
    """

    def __init__(self, X, Y, kern, likelihood, mean_function, name='GPModel'):
        super(GPModel, self).__init__(name=name)
        with tf.variable_scope(self.name):
            self.mean_function = mean_function or Zero()
            self.kern = kern
            self.likelihood = likelihood

        self.X, self.Y = X, Y
        self._parameters = self.mean_function.parameters + self.kern.parameters + self.likelihood.parameters

    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self._build_predict(Xnew)

    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self._build_predict(Xnew, full_cov=True)

    def predict_f_samples(self, Xnew, num_samples):
        """
        Produce samples from the posterior latent function(s) at the points
        Xnew.
        """
        mu, var = self._build_predict(Xnew, full_cov=True)
        jitter = tf.eye(tf.shape(mu)[0], dtype=settings.float_type) * settings.numerics.jitter_level
        samples = []
        for i in range(self.num_latent):
            L = tf.cholesky(var[:, :, i] + jitter)
            shape = tf.stack([tf.shape(L)[0], num_samples])
            V = tf.random_normal(shape, dtype=settings.float_type)
            samples.append(mu[:, i:i + 1] + tf.matmul(L, V))
        return tf.transpose(tf.stack(samples))

    def predict_y(self, Xnew):
        """
        Compute the mean and variance of held-out data at the points Xnew
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var)

    def predict_density(self, Xnew, Ynew):
        """
        Compute the (log) density of the data Ynew at the points Xnew

        Note that this computes the log density of the data individually,
        ignoring correlations between them. The result is a matrix the same
        shape as Ynew containing the log densities.
        """
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        return self.likelihood.predict_density(pred_f_mean, pred_f_var, Ynew)

    @abc.abstractmethod
    def _build_predict(self, *args, **kwargs):
        raise NotImplementedError('') # TODO(@awav): write error message

    def optimize(self, max_iter=1000):
        if not hasattr(self, 'LBFGS-opt'):
            self.LBFGS_opt = LBFGS(tfe.implicit_value_and_gradients(lambda: self.objective), nCorrection=20)
        #print('begin to train this model')
        try:
            self.LBFGS_opt.run()
            fs, gs, xs, vs = zip(*self.LBFGS_opt.history)
            #print([f.numpy() for f in fs])
        except:
         #   import pdb
         #   pdb.set_trace()
            opt = self.LBFGS_opt
            fs, gs, xs, vs = zip(*opt.history)
            print([f.numpy() for f in fs])
            opt.update_vars(vs[0], xs[-3])
            return 

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            for iter in range(2000):
                obj, grads = tfe.implicit_value_and_gradients(lambda: self.objective)()
                optimizer.apply_gradients(grads)

                if iter % 1000 == 0:
                    print('Iter {}: Loss = {}'.format(iter, obj))
        #print('end training')
