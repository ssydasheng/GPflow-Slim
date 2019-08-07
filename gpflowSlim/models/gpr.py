# Copyright 2018 Shengyang Sun
# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
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


from __future__ import absolute_import
import tensorflow as tf

from .. import likelihoods
from .. import settings

from ..decors import name_scope
from ..densities import multivariate_normal, multivariate_normal_feature

from .model import GPModel

class GPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, mean_function=None, obs_var=0.1, num_latent=None, min_var=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        if 'name' in kwargs:
            with tf.variable_scope(kwargs['name']):
                likelihood = likelihoods.Gaussian(var=obs_var, min_var=min_var)
        else:
            likelihood = likelihoods.Gaussian(var=obs_var, min_var=min_var)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_latent = Y.shape[1] if num_latent is None else num_latent

    @name_scope('likelihood')
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        invert_op = getattr(self.kern, 'features', None)
        if invert_op is not None and callable(invert_op):
            m = self.mean_function(self.X)

            return multivariate_normal_feature(self.Y, m, self.kern.features(self.X), self.likelihood.variance)
        else:
            K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
            L = tf.cholesky(K)
            m = self.mean_function(self.X)
            return multivariate_normal(self.Y, m, L)

    @name_scope('predict')
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        invert_op = getattr(self.kern, 'features', None)
        if invert_op is not None and callable(invert_op):
            mX = self.mean_function(self.X)
            m_new = self.mean_function(Xnew)

            feat = self.kern.features(self.X)
            feat_new = self.kern.features(Xnew)

            I_feat = tf.eye(tf.shape(feat)[1], dtype=settings.float_type)
            CtC_I = tf.matmul(tf.transpose(feat), feat) + I_feat * self.likelihood.variance

            ## code to prevent matrix of shape [N_big, N_big]
            #[d, N_train]
            tmp = tf.matmul(
                tf.matmul(feat, feat, transpose_a=True),
                tf.matmul(tf.matrix_inverse(CtC_I), tf.transpose(feat)))
            Ct_CCT_I_inv = (tf.transpose(feat) - tmp) / self.likelihood.variance

            fmean = tf.matmul(feat_new, tf.matmul(Ct_CCT_I_inv, self.Y-mX)) + m_new
            if full_cov:
                BBt = tf.matmul(feat_new, feat_new, transpose_b=True)
                tmp = tf.matmul(
                    feat_new, tf.matmul(tf.matmul(Ct_CCT_I_inv, feat), feat_new, transpose_b=True))
                fvar = BBt - tmp
                shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
                fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
            else:
                # tmp: [N_new, d]
                tmp = tf.matmul(feat_new, tf.matmul(Ct_CCT_I_inv, feat))

                fvar = tf.reduce_sum(feat_new ** 2., -1) - tf.reduce_sum(tmp * feat_new, -1)
                fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        else:
            Kx = self.kern.K(self.X, Xnew)
            K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
            L = tf.cholesky(K)
            A = tf.matrix_triangular_solve(L, Kx, lower=True)
            V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
            fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
            if full_cov:
                fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
                shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
                fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
            else:
                fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
                fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar


class TestPredict(tf.test.TestCase):
    def test_predict(self):
        with self.test_session() as sess:
            Nx, Nn, d = 20, 10, 5
            variance = 2.

            mX = tf.random_normal(shape=[Nx, 1])
            m_new = tf.random_normal(shape=[Nn, 1])
            self.Y = tf.random_normal(shape=[Nx, 1])
            self.X = tf.random_normal(shape=[Nx, d])

            feat = tf.random_normal(shape=[Nx, d])
            feat_new = tf.random_normal(shape=[Nn, d])

            ####################### use feature
            # I = tf.eye(tf.shape(self.X)[0], dtype=settings.float_type)
            I_feat = tf.eye(tf.shape(feat)[1], dtype=settings.float_type)
            CtC_I = tf.matmul(tf.transpose(feat), feat) + I_feat * variance

            # CCt_I_inv = I - tf.matmul(feat, tf.matmul(tf.matrix_inverse(CtC_I), tf.transpose(feat)))
            # CCt_I_inv = CCt_I_inv / self.likelihood.variance
            # Ct_CCT_I_inv = tf.matmul(tf.transpose(feat), CCt_I_inv)

            ## code to prevent matrix of shape [N_big, N_big]
            # [d, N_train]
            tmp = tf.matmul(
                tf.matmul(feat, feat, transpose_a=True),
                tf.matmul(tf.matrix_inverse(CtC_I), tf.transpose(feat)))
            Ct_CCT_I_inv = (tf.transpose(feat) - tmp) / variance

            # B_Ct_CCT_I_inv = tf.matmul(feat_new, Ct_CCT_I_inv)
            # fmean = tf.matmul(B_Ct_CCT_I_inv, self.Y-mX) + m_new

            fmean = tf.matmul(feat_new, tf.matmul(Ct_CCT_I_inv, self.Y - mX)) + m_new

            BBt_full = tf.matmul(feat_new, feat_new, transpose_b=True)
            tmp_full = tf.matmul(
                feat_new, tf.matmul(tf.matmul(Ct_CCT_I_inv, feat), feat_new, transpose_b=True))
            fvar_full = BBt_full - tmp_full
            shape_full = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar_full = tf.tile(tf.expand_dims(fvar_full, 2), shape_full)

            # BCt = tf.transpose(tf.matmul(feat, feat_new, transpose_b=True))
            # tmp: [N_new, d]
            tmp_diag = tf.matmul(
                feat_new, tf.matmul(Ct_CCT_I_inv, feat))

            fvar_diag = tf.reduce_sum(feat_new ** 2., -1) - tf.reduce_sum(tmp_diag * feat_new, -1)
            fvar_diag = tf.tile(tf.reshape(fvar_diag, (-1, 1)), [1, tf.shape(self.Y)[1]])

            ############################ standard one
            Kx = tf.matmul(feat, feat_new, transpose_b=True)
            K = tf.matmul(feat, feat, transpose_b=True) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * variance
            L = tf.cholesky(K)
            A = tf.matrix_triangular_solve(L, Kx, lower=True)
            V = tf.matrix_triangular_solve(L, self.Y - mX)
            standard_fmean = tf.matmul(A, V, transpose_a=True) + m_new

            standard_fvar_full = tf.matmul(feat_new, feat_new, transpose_b=True) - tf.matmul(A, A, transpose_a=True)
            standard_shape_full = tf.stack([1, 1, tf.shape(self.Y)[1]])
            standard_fvar_full = tf.tile(tf.expand_dims(standard_fvar_full, 2), standard_shape_full)

            standard_fvar_diag = tf.matrix_diag_part(tf.matmul(feat_new, feat_new, transpose_b=True)) - tf.reduce_sum(tf.square(A), 0)
            standard_fvar_diag = tf.tile(tf.reshape(standard_fvar_diag, (-1, 1)), [1, tf.shape(self.Y)[1]])

            a, b, c, d, e, f = sess.run([fmean, standard_fmean, fvar_full, standard_fvar_full, fvar_diag, standard_fvar_diag])
            self.assertAllClose(a, b, rtol=1e-4, atol=1e-4)
            self.assertAllClose(c, d, rtol=1e-4, atol=1e-4)
            self.assertAllClose(e, f, rtol=1e-4, atol=1e-4)
