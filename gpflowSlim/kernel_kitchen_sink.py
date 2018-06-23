from __future__ import print_function, absolute_import
from functools import reduce
import warnings

import tensorflow as tf
import numpy as np
from scipy.linalg import qr
from abc import abstractmethod

from .kernels import Kernel
from . import transforms
from .params import Parameter
from . import settings


class Sampler(object):
    def __init__(self, input_dim, n_components):
        self.input_dim = input_dim
        self.n_components = n_components

    def check_dim(self, X):
        assert_dim = tf.assert_equal(
            tf.shape(X)[1], self.input_dim,
            message=['input dimension not compatible with the init value'])
        return assert_dim

    @abstractmethod
    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        pass


class RBFSampler(Sampler):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    It implements a variant of Random Kitchen Sinks.[1]
    Read more in the :ref:`User Guide <rbf_kernel_approx>`.
    Parameters
    ----------
    input_dim: int
        Input dimension.
    ls : float
        Parameter of RBF kernel: exp(-x^2 / (2 * ls^2))
    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (http://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
    """

    def __init__(self, input_dim, ls=1., var=1., n_components=100, scope='RBFSampler'):

        with tf.variable_scope(scope):
            self._ls = Parameter(ls, transform=transforms.positive,
                                name='ls', dtype=settings.float_type)
            self._variance = Parameter(var, transform=transforms.positive,
                                       name='variance', dtype=settings.float_type)

        self.random_weights_ =  tf.cast(np.random.normal(size=(input_dim, n_components))
                                / tf.reshape(self.ls, [-1, 1]), settings.tf_float)
        self.random_offset_ = np.random.uniform(
            0, 2 * np.pi, size=n_components).astype(settings.float_type)
        super(RBFSampler, self).__init__(input_dim, n_components)

    @property
    def ls(self):
        return self._ls.value

    @property
    def variance(self):
        return self._variance.value

    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        with tf.control_dependencies([self.check_dim(X)]):
            projection = tf.matmul(X, self.random_weights_) + self.random_offset_
            feature = tf.cos(projection) * tf.cast(tf.sqrt(2. / self.n_components), settings.tf_float)

        return feature * (self.variance ** 0.5)


class CosineRBFSampler(Sampler):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    It implements a variant of Random Kitchen Sinks.[1]
    Read more in the :ref:`User Guide <rbf_kernel_approx>`.
    Parameters
    ----------
    input_dim: int
        Input dimension.
    ls : float
        Parameter of RBF kernel: exp(-x^2 / (2 * ls^2))
    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (http://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
    """

    def __init__(self, input_dim, ls=1., var=1., n_components=100, scope='RBFSampler'):

        with tf.variable_scope(scope):
            self._ls = Parameter(ls, transform=transforms.positive,
                                name='ls', dtype=settings.float_type)
            self._variance = Parameter(var, transform=transforms.positive,
                                       name='variance', dtype=settings.float_type)
            self._cos_ls = Parameter(np.ones([input_dim]),
                                     transform=transforms.positive,
                                    name='cos_ls', dtype=settings.float_type)

        self.random_weights_ = tf.cast(np.random.normal(size=(input_dim, n_components))
                                / tf.reshape(self.ls, [-1, 1]), settings.tf_float)
        self.random_offset_ = np.random.uniform(
            0, 2 * np.pi, size=n_components).astype(settings.float_type)
        super(CosineRBFSampler, self).__init__(input_dim, n_components)

    @property
    def ls(self):
        return self._ls.value

    @property
    def cos_ls(self):
        return self._cos_ls.value

    @property
    def variance(self):
        return self._variance.value

    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """

        with tf.control_dependencies([self.check_dim(X)]):
            projection = tf.matmul(tf.cos(X / self.cos_ls), self.random_weights_) + self.random_offset_
            feature = tf.cos(projection) * tf.cast(tf.sqrt(2. / self.n_components), settings.tf_float)

        return feature * (self.variance ** 0.5)


class LinearSampler(Sampler):
    def __init__(self, input_dim, var=1., n_components=None, scope='LinearSampler'):
        with tf.variable_scope(scope):
            self._variance = Parameter(var, transform=transforms.positive,
                                       name='variance', dtype=settings.float_type)
        super(LinearSampler, self).__init__(input_dim, n_components or input_dim)

    @property
    def variance(self):
        return self._variance.value

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            X_tile = tf.concat([X for _ in range(int(np.ceil(self.n_components / self.input_dim)))], axis=-1)
            return X_tile[:, :self.n_components] * (self.variance * self.input_dim / float(self.n_components))** 0.5


class CosineSampler(Sampler):
    def __init__(self, input_dim, ls=1., var=1., n_components=2, scope='CosineSampler'):
        with tf.variable_scope(scope):
            self._ls = Parameter(
                ls, transform=transforms.positive, name='ls', dtype=settings.float_type)
            self._variance = Parameter(
                var, transform=transforms.positive, name='variance', dtype=settings.float_type)
            self._weights = Parameter(
                np.random.normal(size=[input_dim, 1]).astype(settings.float_type),
                name='weights')
        super(CosineSampler, self).__init__(input_dim, n_components)

    @property
    def variance(self):
        return self._variance.value

    @property
    def ls(self):
        return self._ls.value

    @property
    def weights(self):
        return self._weights.value

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            # mul: [None, 1]
            mul = tf.matmul(X / self.ls, self.weights)
            feat = tf.concat([tf.cos(mul), tf.sin(mul)], axis=-1)
            feat = tf.concat([feat for _ in range(int(np.ceil(self.n_components / 2)))], axis=-1)
        return feat[:, :self.n_components] * (self.variance * 2. / self.n_components)** 0.5


class CosineV2Sampler(Sampler):
    def __init__(self, input_dim, ls=1., var=1., n_components=10, scope='CosineSampler'):
        with tf.variable_scope(scope):
            self._ls = Parameter(
                ls, transform=transforms.positive, name='ls', dtype=settings.float_type)
            self._variance = Parameter(
                var, transform=transforms.positive, name='variance', dtype=settings.float_type)
            self._weights = Parameter(
                np.random.normal(size=[input_dim, 1]).astype(settings.float_type),
                name='weights')
            self.bias = np.random.uniform(0, 2*np.pi, size=[n_components, 1])
        super(CosineV2Sampler, self).__init__(input_dim, n_components)

    @property
    def variance(self):
        return self._variance.value

    @property
    def ls(self):
        return self._ls.value

    @property
    def weights(self):
        return self._weights.value

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            # mul: [None, 1]
            mul = tf.matmul(X / self.ls, self.weights)
            feat = tf.concat([tf.cos(mul+self.bias[i]) for i in range(self.n_components)], axis=-1)

        return feat * (self.variance * 2. / self.n_components)** 0.5


class ArcCosineSampler(Sampler):
    def __init__(self, input_dim, var=1., p=1, n_components=100, scope='ArcCosineSampler'):
        self.p = p
        with tf.variable_scope(scope):
            self._variance = Parameter(var, transform=transforms.positive,
                                       name='variance', dtype=settings.float_type)

        self.random_weights_ = np.random.normal(
            size=(input_dim, n_components)).astype(settings.float_type)
        self.random_bias = np.random.normal(
            size=(1, n_components)).astype(settings.float_type)
        super(ArcCosineSampler, self).__init__(input_dim, n_components)

    @property
    def variance(self):
        return self._variance.value

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            projection = tf.matmul(X, self.random_weights_) + self.random_bias
            Heaviside = tf.cast(tf.greater(projection, 0), settings.tf_float)
            feature = tf.pow(projection, self.p) * Heaviside

        tmp = tf.cast(tf.sqrt(2. / self.n_components), settings.tf_float)
        return feature * (self.variance ** 0.5) * tmp


class ConstantSampler(Sampler):
    def __init__(self, input_dim, var=1., n_components=1, scope='ConstantSampler'):
        with tf.variable_scope(scope):
            self._variance = Parameter(var, transform=transforms.positive,
                                       name='variance', dtype=settings.float_type)
        super(ConstantSampler, self).__init__(input_dim, n_components)

    @property
    def variance(self):
        return self._variance.value

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            feat = tf.ones([tf.shape(X)[0], self.n_components], dtype=settings.tf_float)
        return feat * (self.variance / self.n_components) ** 0.5


class EqApproxSumSampler(Sampler):
    """
    :param samplers: list of samplers. They are of the same n_components
    :param weights: list of tensor of shape []. Coefficients on these samplers.
    """
    def __init__(self, samplers, weights, n_components):
        for sampler in samplers:
            assert isinstance(sampler, Sampler)
            assert sampler.input_dim == samplers[0].input_dim
        assert len(samplers) == len(weights)
        assert len(samplers) >= 2

        self.samplers = samplers
        self.weights = weights

        ns = [n_components // (len(self.samplers)) for _ in range(len(self.samplers) - 1)]
        self.ns = ns + [n_components - sum(ns)]
        self.inds = [np.random.permutation(s.n_components)[:n] for n, s in zip(self.ns, self.samplers)]

        super(EqApproxSumSampler, self).__init__(
            samplers[0].input_dim,
            n_components)

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            feats = [tf.gather(sampler.transform(X), ind, axis=1) for ind, sampler in zip(self.inds, self.samplers)]
            feats = [feat * (w * s.n_components / n) ** 0.5
                     for n, w, feat, s in zip(self.ns, self.weights, feats, self.samplers)]
            feat = tf.concat(feats, axis=-1)
        return feat


class NEqApproxSumSampler(Sampler):
    """
    :param samplers: list of samplers. They are of the same n_components
    :param weights: list of tensor of shape []. Coefficients on these samplers.
    """
    def __init__(self, samplers, weights, n_components):
        for sampler in samplers:
            assert isinstance(sampler, Sampler)
            assert sampler.input_dim == samplers[0].input_dim
        assert len(samplers) >= 2
        assert sum([s.n_components for s in samplers]) >= n_components

        assert_equal = tf.assert_equal(len(samplers), tf.shape(weights)[0],
                                       message='length of samplers and weights not equal')
        with tf.control_dependencies([assert_equal]):
            average = n_components // len(samplers)
        # count how many dimensions does each sampler need
        self.ns = [None for _ in range(len(samplers))]
        # which dimensions does each sampler need
        self.inds = [None for _ in range(len(samplers))]

        # count all the samplers whose n_component is smaller than average
        sum_, n_ = 0, 0
        for i in range(len(samplers)):
            if samplers[i].n_components <= average:
                self.ns[i] = samplers[i].n_components
                self.inds[i] = list(range(samplers[i].n_components))
                sum_ = sum_ + samplers[i].n_components
                n_ = n_ + 1

        # give other samplers equal dimensions
        if n_ < len(samplers):
            average = (n_components - sum_) // (len(samplers) - n_)
            other_ns = [average for _ in range(len(samplers) - n_ - 1)]
            other_ns = other_ns + [n_components - sum_ - sum(other_ns)]

            # write ns, inds for other samplers
            j = 0
            for i in range(len(samplers)):
                if self.ns[i] is not None:
                    continue
                self.ns[i] = other_ns[j]
                self.inds[i] = np.random.permutation(samplers[i].n_components)[:other_ns[j]]
                j = j + 1
            assert j == len(other_ns)

        self.samplers = samplers
        self.weights = weights

        super(NEqApproxSumSampler, self).__init__(
            samplers[0].input_dim,
            n_components)

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            feats = [tf.gather(sampler.transform(X), ind, axis=1)
                     for ind, sampler in zip(self.inds, self.samplers)]
            feats = [feat * (self.weights[i] * s.n_components / n) ** 0.5
                     for i, (n, feat, s) in enumerate(zip(self.ns, feats, self.samplers))]
            feat = tf.concat(feats, axis=-1)
        return feat


class ApproxProdSampler(Sampler):
    def __init__(self, samplers, n_components):
        for sampler in samplers:
            assert isinstance(sampler, Sampler)
            assert sampler.input_dim == samplers[0].input_dim
        assert len(samplers) >= 2

        self.samplers = samplers
        self.mixtures = [self._weights(n_components, sampler.n_components)
                         for sampler in self.samplers]

        super(ApproxProdSampler, self).__init__(
            self.samplers[0].input_dim,
            n_components)

    @abstractmethod
    def _weights(self, n, d):
        pass

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            feats = [
                sampler.transform(X) *
                tf.cast(tf.to_float(sampler.n_components), settings.tf_float) ** 0.5
                for sampler in self.samplers]
            mixture_prod = 1.
            for feat, mixture in zip(feats, self.mixtures):
                # if settings.float_type == 'float32':
                #     mixture_prod = mixture_prod * tf.matmul(
                #         feat, tf.transpose(mixture),
                #         b_is_sparse=isinstance(self, SubsetApproxProdSampler))
                # else:
                #     mixture_prod = mixture_prod * tf.cast(tf.matmul(
                #         tf.cast(feat, tf.float32), tf.cast(tf.transpose(mixture), tf.float32),
                #         b_is_sparse=isinstance(self, SubsetApproxProdSampler)), tf.float64)
                if isinstance(self, SubsetApproxProdSampler):
                    tmp = tf.transpose(tf.sparse_tensor_dense_matmul(mixture, tf.transpose(feat)))
                else:
                    tmp = tf.matmul(feat, tf.transpose(mixture))
                mixture_prod = mixture_prod * tmp

        return  mixture_prod /  tf.cast(tf.sqrt(tf.to_float(self.n_components)), settings.tf_float)


def _gen_random(n, d):
    mat = np.random.uniform(
        low=-1, high=1,
        size=[n, d]).astype(settings.float_type)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)

def _gen_orthogonal(n, d):
    reps = int(np.ceil(n / d))
    Q = np.empty((d*reps, d))

    for r in range(reps):
        W = np.random.randn(d, d)
        Q[(r * d):((r + 1) * d), :] = qr(W)[0].transpose()
    return Q[:n].astype(settings.float_type)

def _gen_identity(n, d):
    # reps = int(np.ceil(n / d))
    # Q = np.empty((d*reps, d))
    #
    # for r in range(reps):
    #     Q[(r * d):((r + 1) * d), :] = np.eye(d)
    # permutation = np.random.permutation(d*reps)
    # return Q[permutation[:n]].astype(settings.float_type)
    # #
    indices = np.concatenate([np.expand_dims(range(n), 1), np.random.randint(d, size=[n, 1])], axis=1)
    values = np.ones([n], dtype=settings.tf_float)
    return tf.SparseTensor(indices, values, dense_shape=[n, d])

def _gen_one(n, d):
    mat = np.random.choice([-1., 1.], size=[n, d])
    return mat / d**0.5


class RandomApproxProdSampler(ApproxProdSampler):
    def _weights(self, n, d):
        return _gen_random(n, d).astype(settings.float_type)

class OrthogonalApproxProdSampler(ApproxProdSampler):
    def _weights(self, n, d):
        return _gen_orthogonal(n, d).astype(settings.float_type)

class SubsetApproxProdSampler(ApproxProdSampler):
    def _weights(self, n, d):
        return _gen_identity(n, d)

class OneApproxProdSampler(ApproxProdSampler):
    def _weights(self, n, d):
        return _gen_one(n, d).astype(settings.float_type)

def _generate_sketch_matrix(rand_h, rand_s, output_dim):
    """
    Return a sparse matrix used for tensor sketch operation in compact bilinear
    pooling
    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.
    Returns:
        a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
    """

    # Generate a sparse matrix for tensor count sketch
    rand_h = rand_h.astype(settings.int_type)
    rand_s = rand_s.astype(settings.float_type)
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1).astype(settings.float_type)
    sparse_sketch_matrix = tf.sparse_reorder(
        tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    return sparse_sketch_matrix

class SketchApproxProdSampler(ApproxProdSampler):
    def __init__(self, samplers, n_components):
        for sampler in samplers:
            assert isinstance(sampler, Sampler)
            assert sampler.input_dim == samplers[0].input_dim
        assert len(samplers) >= 2

        self.samplers = samplers
        self.hs = [self._h(sampler.n_components, n_components) for sampler in self.samplers]
        self.ss = [self._s(sampler.n_components, n_components) for sampler in self.samplers]

        self.sparse_sketch_matrix = [_generate_sketch_matrix(h, s, n_components)
                                     for h, s in zip(self.hs, self.ss)]

        self.input_dim = self.samplers[0].input_dim
        self.n_components = n_components

    def _h(self, dim, n_range):
        return np.random.randint(n_range, size=[dim]).astype(settings.int_type)

    def _s(self, dim, n_range):
        return np.random.choice([-1, 1], size=[dim]).astype(settings.float_type)

    # @staticmethod
    # def _sketch_operator(v, h, s, n_range):
    #     """
    #     :param v: Tensor of shape [batch_size, dim]
    #     :return: Tensor of shape [batch_size, n_range]
    #     """
    #     tile_v = tf.tile(tf.expand_dims(v, 1), [1, n_range, 1])
    #     tile_s = tf.expand_dims(tf.expand_dims(s, 0), 0)
    #     tile_h = tf.tile(tf.expand_dims(h, 0), [n_range, 1])
    #     condition = tf.cast(tf.equal(
    #         tile_h, tf.expand_dims(tf.range(n_range, dtype=settings.tf_int), [1])), settings.tf_float)
    #
    #     return tf.reduce_sum(tile_v * tile_s * condition, -1)
    #
    # @staticmethod
    # def _circular_conv(a, b):
    #     a_fft = tf.spectral.rfft(tf.cast(a, tf.float32))
    #     b_fft = tf.spectral.rfft(tf.cast(b, tf.float32))
    #     return tf.cast(tf.spectral.irfft(a_fft * b_fft), settings.tf_float)

    def transform(self, X):
        with tf.control_dependencies([self.check_dim(X)]):
            ffts = []
            for sparse_sketch_matrix, sampler in zip(self.sparse_sketch_matrix, self.samplers):
                feat = sampler.transform(X)
                sketch = tf.transpose(tf.sparse_tensor_dense_matmul(
                    sparse_sketch_matrix, feat, adjoint_a=True, adjoint_b=True))

                fft = tf.fft(tf.cast(tf.complex(
                    real=sketch, imag=tf.zeros_like(sketch, dtype=settings.tf_float)), tf.complex64))
                ffts.append(fft)

            fft_prod = tf.reduce_prod(ffts, axis=0)
            result = tf.cast(tf.real(tf.ifft(fft_prod)), settings.float_type)
        return result


class OuterProductSampler(Sampler):
    def __init__(self, sampler1, sampler2):
        assert isinstance(sampler1, Sampler)
        assert isinstance(sampler2, Sampler)
        assert sampler1.input_dim == sampler2.input_dim

        self.sampler1 = sampler1
        self.sampler2 = sampler2
        super(OuterProductSampler, self).__init__(
            sampler1.input_dim,
            sampler1.n_components * sampler2.n_components)

    def transform(self, X):
        with tf.control_dependencies([self.sampler1.check_dim(X)]):
            feat1 = self.sampler1.transform(X)
            feat2 = self.sampler2.transform(X)
            tile_feat1 = tf.tile(feat1, [1, tf.shape(feat2)[1]])

            repeat_feat2 = tf.tile(tf.expand_dims(feat2, -1), [1, 1, tf.shape(feat1)[1]])
            repeat_feat2 = tf.reshape(repeat_feat2, [tf.shape(repeat_feat2)[0], -1])

        return tile_feat1 * repeat_feat2


class RFFSampler(Sampler):
    def transform(self, X):
        projection = tf.matmul(X, self.random_weights_) + self.random_offset_
        feature = tf.cos(projection) * tf.cast(tf.sqrt(2. / self.n_components), settings.tf_float)

        return feature * (self.variance ** 0.5)


class RFFApproxSumSampler(RFFSampler):
    def __init__(self, samplers, n_components):
        for sampler in samplers:
            assert isinstance(sampler, Sampler)
            assert sampler.input_dim == samplers[0].input_dim

        sum_var = tf.reduce_sum([s.variance for s in samplers])

        weights = []
        for sampler in samplers:
            i0 = tf.constant(0.)
            w0 = tf.transpose(tf.random_shuffle(tf.transpose(sampler.random_weights_)))
            body = lambda i, w: [
                i+1,
                tf.concat([w, tf.transpose(tf.random_shuffle(tf.transpose(sampler.random_weights_)))], axis=1)]

            condition = lambda i, w: ((i+1) * sampler.n_components) < (n_components * sampler.variance / sum_var)
            shape_ = [i0.get_shape(), tf.TensorShape([w0.get_shape().as_list()[0], None])]
            _, ws = tf.while_loop(
                condition, body,
                loop_vars=[i0, w0],
                shape_invariants=shape_
            )

            weights.append(tf.concat(ws, axis=1))
        self.random_weights_ = tf.concat(weights, axis=1)[:, :n_components]

        self.random_offset_ = np.random.uniform(
            0, 2 * np.pi, size=n_components).astype(settings.float_type)
        self.variance = sum_var

        super(RFFApproxSumSampler, self).__init__(
            samplers[0].input_dim,
            n_components)


class RFFApproxProdSampler(RFFSampler):
    def __init__(self, samplers, n_components):
        for sampler in samplers:
            assert isinstance(sampler, Sampler)
            assert sampler.input_dim == samplers[0].input_dim

        weights = 0.
        for sampler in samplers:
            reps = int(np.ceil(n_components / sampler.n_components))
            weight = tf.concat([tf.transpose(tf.random_shuffle(tf.transpose(sampler.random_weights_), seed=i*666))
                                for i in range(reps)], axis=1)[:, :n_components]
            weights = weights + weight
        self.random_weights_ = weights
        self.random_offset_ = np.random.uniform(
            0, 2 * np.pi, size=n_components).astype(settings.float_type)

        self.variance = np.prod([s.variance for s in samplers])
        super(RFFApproxProdSampler, self).__init__(
            samplers[0].input_dim,
            n_components)


class TransformSampler(Sampler):
    def __init__(self, input_dim, n_components, transform):
        self._transform = transform
        super(TransformSampler, self).__init__(input_dim, n_components)

    def transform(self, X):
        return self._transform(X)

################## ################## ################## ################## ################## ##################

class SamplerKernel(Kernel):
    def __init__(self, sampler):
        self.sampler = sampler
        super(SamplerKernel, self).__init__(input_dim=sampler.input_dim)

    def K(self, X, X2=None, presliced=False):
        feat1 = self.sampler.transform(X)
        feat2 = self.sampler.transform(X if X2 is None else X2)
        return tf.matmul(feat1, tf.transpose(feat2))

    def Kdiag(self, X, presliced=False):
        feat1 = self.sampler.transform(X)
        return tf.reduce_sum(feat1 ** 2., -1)

    def features(self, X):
        return self.sampler.transform(X)

################## ################## ################## ################## ################## ##################

class SamplerGroup(object):
    def __init__(self, n):
        self.n_samplers = n

    @abstractmethod
    def transform(self, X):
        """
        :param X: tensor of shape [batch_size, d]
        :return: list of features of shape [batch_size, d']
        """
        pass

    def __len__(self):
        return self.n_samplers

    def length(self, ind):
        pass

    @property
    def input_dim(self):
        raise NotImplementedError

    @abstractmethod
    def elements(self):
        pass


class ListSamplerGroup(SamplerGroup):
    def __init__(self, samplers):
        super(ListSamplerGroup, self).__init__(len(samplers))
        for ss in samplers:
            assert isinstance(ss, Sampler)
        self.samplers = samplers

    def transform(self, X):
        return [sampler.transform(X) for sampler in self.samplers]

    def length(self, ind):
        assert ind >= 0
        assert ind < len(self)
        return self.samplers[ind].n_components

    @property
    def input_dim(self):
        return self.samplers[0].input_dim

    def elements(self):
        return self.samplers


class FullyConnectedSamplerGroup(SamplerGroup):
    def __init__(self, input_group, weights, n_samplers, n_components):
        assert isinstance(input_group, SamplerGroup)
        self.input_group = input_group
        self.weights = weights
        self.n_components = n_components
        self.n_samplers = n_samplers

        ################## compute n for each sampler in input_group ##################
        average = n_components // len(input_group)
        self.ns = [None for _ in range(len(input_group))]

        sum_, n_ = 0, 0
        for i in range(len(input_group)):
            if input_group.length(i) <= average:
                self.ns[i] = input_group.length(i)
                sum_ = sum_ + input_group.length(i)
                n_ = n_ + 1

        if n_ < len(input_group):
            average = (n_components - sum_) // (len(input_group) - n_)
            other_ns = [average for _ in range(len(input_group) - n_ - 1)]
            other_ns = other_ns + [n_components - sum_ - sum(other_ns)]

            # write ns, inds for other samplers
            j = 0
            for i in range(len(input_group)):
                if self.ns[i] is not None:
                    continue
                self.ns[i] = other_ns[j]
                j = j + 1
            assert j == len(other_ns)
        ####################### Assign random index for these samplers #############################
        self.indices = [[np.random.permutation(self.input_group.length(j))[:n]
                         for j, n in enumerate(self.ns)]
                        for _ in range(self.n_samplers)]

        super(FullyConnectedSamplerGroup, self).__init__(n_samplers)

    def length(self, ind):
        assert ind >= 0
        assert ind < len(self)
        return self.n_components

    def transform(self, X):
        feats = self.input_group.transform(X)
        output_feats = []
        for i in range(self.n_samplers):
            f = [tf.gather(feat, ind, axis=1) for feat, ind in zip(feats, self.indices[i])]
            f = [ff * (self.weights[i][j] * self.input_group.length(j) / float(n))**0.5
                 for j, (ff, n) in enumerate(zip(f, self.ns))]
            output_feats.append(tf.concat(f, axis=-1))
        return output_feats

    @property
    def input_dim(self):
        return self.input_group.input_dim

    def elements(self):
        def _feature(idx, X):
            feats = self.input_group.transform(X)
            f = [tf.gather(feat, ind, axis=1) for feat, ind in zip(feats, self.indices[idx])]
            f = [ff * (self.weights[idx][j] * self.input_group.length(j) / float(n)) ** 0.5
                 for j, (ff, n) in enumerate(zip(f, self.ns))]
            return tf.concat(f, axis=-1)

        return [TransformSampler(self.input_dim, self.length(i), lambda X: _feature(i, X))
                for i in range(self.n_samplers)]


class ReduceProdSamplerGroup(SamplerGroup):
    def __init__(self, input_group, step, n_components, product_type):
        assert len(input_group) % step == 0
        assert isinstance(input_group, SamplerGroup)
        assert product_type in ['random', 'subset', 'orthogonal', 'one']

        super(ReduceProdSamplerGroup, self).__init__(len(input_group) // step)
        self.n_components = n_components
        self.product_type = product_type
        self.step = step
        self.input_group = input_group
        self.weights = {'random': _gen_random,
                        'subset': _gen_identity,
                        'orthogonal': _gen_orthogonal,
                        'one': _gen_one}.get(product_type)
        self.mixtures = [[
            self.weights(n_components, input_group.length(step*i+j))
            for j in range(step)]
            for i in range(len(input_group) // step)
        ]

    def length(self, ind):
        assert ind >= 0
        assert ind < len(self)
        return self.n_components

    def transform(self, X):
        feats = self.input_group.transform(X)
        output_feats = []
        for i in range(len(self.input_group) // self.step):
            mixture_prod = 1
            for j in range(self.step):
                f = feats[self.step*i+j] * tf.cast(self.input_group.length(self.step*i+j), settings.tf_float)**0.5

                # if settings.float_type == 'float32':
                #     mixture_prod = mixture_prod * tf.matmul(
                #         f, tf.transpose(self.mixtures[i][j]),
                #         b_is_sparse=self.product_type == 'subset')
                # else:
                #     mixture_prod = mixture_prod * tf.cast(tf.matmul(
                #         tf.cast(f, tf.float32), tf.cast(tf.transpose(self.mixtures[i][j]), tf.float32),
                #         b_is_sparse=self.product_type == 'subset'), tf.float64)

                if self.product_type == 'subset':
                    tmp = tf.transpose(tf.sparse_tensor_dense_matmul(self.mixtures[i][j], tf.transpose(f)))
                else:
                    tmp = tf.matmul(f, tf.transpose(self.mixtures[i][j]))
                mixture_prod = mixture_prod * tmp
            output_feats.append(mixture_prod / tf.cast(self.n_components, settings.tf_float)**0.5)
        return output_feats

    @property
    def input_dim(self):
        return self.input_group.input_dim

    def elements(self):
        def _feature(idx, X):
            feats = self.input_group.transform(X)
            mixture_prod = 1
            for j in range(self.step):
                f = feats[self.step * idx + j] * tf.cast(self.input_group.length(self.step * idx + j),
                                                       settings.tf_float) ** 0.5
                mixture_prod = mixture_prod * tf.matmul(f, tf.transpose(self.mixtures[idx][j]))
            return mixture_prod / tf.cast(self.n_components, settings.tf_float) ** 0.5

        return [TransformSampler(self.input_dim, self.length(i), lambda X: _feature(i, X))
                for i in range(self.n_samplers)]

class ConcatSamplerGroup(SamplerGroup):
    def __init__(self, group1, group2):
        assert isinstance(group1, SamplerGroup)
        assert isinstance(group2, SamplerGroup)

        self.group1=group1
        self.group2=group2

        super(ConcatSamplerGroup, self).__init__(len(group1) + len(group2))

    def transform(self, X):
        return self.group1.transform(X) + self.group2.transform(X)

    def length(self, ind):
        assert ind >= 0
        assert ind < len(self)
        if ind < len(self.group1):
            return self.group1.length(ind)
        return self.group2.length(ind-len(self.group1))

    @property
    def input_dim(self):
        return self.group1.input_dim

    def elements(self):
        return self.group1.elements() + self.group2.elements()

################## ################## ################## ################## ################## ##################

class SamplerGroupKernel(Kernel):
    def __init__(self, sampler):
        assert len(sampler) == 1
        self.sampler = sampler
        super(SamplerGroupKernel, self).__init__(input_dim=sampler.input_dim)

    def K(self, X, X2=None, presliced=False):
        feat1 = self.sampler.transform(X)[0]
        if X2 is None:
            feat2 = feat1
        else:
            feat2 = self.sampler.transform(X2)[0]
        return tf.matmul(feat1, tf.transpose(feat2))

    def Kdiag(self, X, presliced=False):
        feat1 = self.sampler.transform(X)[0]
        return tf.reduce_sum(feat1 ** 2., -1)

    def features(self, X):
        return self.sampler.transform(X)[0]