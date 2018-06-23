import tensorflow as tf
from scipy.sparse.linalg import eigsh
import numpy as np

from . import settings


def vectorize(mat):
    return tf.reshape(tf.transpose(mat), [-1, 1])

def devectorize(vec, shape):
    return tf.transpose(tf.reshape(vec, shape[::-1]))

def batch_vectorize(mat):
    return tf.reshape(tf.transpose(mat), [-1, tf.shape(mat)[-1]])

def batch_devectorize(vec, shape):
    return tf.transpose(tf.reshape(vec, shape[::-1]))

class Kronecker(object):

    def __init__(self, vars):
        self.vars = vars
        self.sizes = [tf.shape(v)[0] for v in self.vars]

    def matrix_inverse_root(self, k=10):
        roots = []
        for v in self.vars:
            v = v.numpy() if isinstance(v, tf.Tensor) else v
            eigvals, eigvecs = eigsh(v, k=min(k, v.shape[0]-1))
            eigvals = np.maximum(eigvals, 1e-10)
            eigvals = tf.diag(1. / eigvals ** 0.5)
            root = tf.matmul(eigvecs, tf.matmul(eigvals, eigvecs.transpose()))
            roots.append(root)
        return self.__class__(roots)

    def matrix_root(self, k=10):
        roots = []
        for v in self.vars:
            v = v.numpy() if isinstance(v, tf.Tensor) else v
            eigvals, eigvecs = eigsh(v, k=min(k, v.shape[0]-1))
            eigvals = np.maximum(eigvals, 1e-10)
            eigvals = tf.diag(eigvals ** 0.5)
            root = tf.matmul(eigvecs, tf.matmul(eigvals, eigvecs.transpose()))
            roots.append(root)
        return self.__class__(roots)

    def matmul_vec(self, vec):
        res = vec
        for v in self.vars:
            res = tf.transpose(devectorize(res, [-1, tf.shape(v)[0]]))
            factor = tf.matmul(v, res)
            res = tf.reshape(tf.transpose(factor), [-1, 1])
        res = devectorize(res, [-1, tf.shape(vec)[-1]])
        return res

class testKronecker(tf.test.TestCase):
    def test_2elements(self):
        with self.test_session() as sess:
            var1 = tf.random_normal(shape=[4, 4])
            var2 = tf.random_normal(shape=[3, 3])
            var1 = tf.matmul(var1, var1, transpose_b=True)
            var2 = tf.matmul(var2, var2, transpose_b=True)
            kron = Kronecker([var1, var2])

            v = tf.random_normal(shape=[12, 1])
            v_mat = devectorize(v, [3, 4])

            kv = kron.matmul_vec(v)
            kp = tf.contrib.kfac.utils.kronecker_product
            real_kv = tf.matmul(kp(var1, var2), v)
            # real_kv = vectorize(tf.matmul(tf.matmul(var2, v_mat), var1, transpose_b=True))

            aa, bb = sess.run([kv, real_kv])
            print(aa)
            print(bb)
            self.assertAllClose(aa, bb)

    def test_3elements(self):
        with self.test_session() as sess:
            var1 = tf.random_normal(shape=[3, 3])
            var2 = tf.random_normal(shape=[4, 4])
            var3 = tf.random_normal(shape=[5, 5])
            var1 = tf.matmul(var1, var1, transpose_b=True)
            var2 = tf.matmul(var2, var2, transpose_b=True)
            var3 = tf.matmul(var3, var3, transpose_b=True)
            kron = Kronecker([var1, var2, var3])

            v = tf.random_normal(shape=[12 * 5, 1])
            kv = kron.matmul_vec(v)

            kp = tf.contrib.kfac.utils.kronecker_product
            M = kp(var1, kp(var2, var3))
            real_kv = tf.matmul(M, v)

            aa, bb = sess.run([kv, real_kv])
            print(aa.squeeze())
            print(bb.squeeze())
            self.assertAllClose(aa, bb, rtol=1e-5, atol=1e-4)

    def test_3elements_batch(self):
        with self.test_session() as sess:
            var1 = tf.random_normal(shape=[3, 3])
            var2 = tf.random_normal(shape=[4, 4])
            var3 = tf.random_normal(shape=[5, 5])
            var1 = tf.matmul(var1, var1, transpose_b=True)
            var2 = tf.matmul(var2, var2, transpose_b=True)
            var3 = tf.matmul(var3, var3, transpose_b=True)
            kron = Kronecker([var1, var2, var3])

            v = tf.random_normal(shape=[12 * 5, 5])
            kv = kron.matmul_vec(v)

            kp = tf.contrib.kfac.utils.kronecker_product
            M = kp(var1, kp(var2, var3))
            real_kv = tf.matmul(M, v)

            aa, bb = sess.run([kv, real_kv])
            print(aa.squeeze())
            print('------------------')
            print(bb.squeeze())
            self.assertAllClose(aa, bb, rtol=1e-5, atol=1e-4)
