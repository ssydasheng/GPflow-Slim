# Copyright 2018 Shengyang Sun
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
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from .kernel_kitchen_sink import *
import gpflowSlim as gpf

# class TestRBFSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             np.random.seed(123)
#             gamma = np.array([1., 2.])
#             sampler = RBFSampler(2, ls=gamma, n_components=500)
#             kernel_approx = SamplerKernel(sampler)
#             X = np.random.normal(size=[4, 2]).astype(gpf.settings.float_type)
#
#             rbf_kernel = gpf.kernels.RBF(input_dim=2, lengthscales=gamma)
#             sess.run(tf.global_variables_initializer())
#             # print(rbf_kernel.K(X).eval())
#             # print(kernel_approx.K(X).eval())
#             self.assertAllClose(
#                 rbf_kernel.K(X).eval(), kernel_approx.K(X).eval(),
#                 rtol=5e-2, atol=5e-2)
#
#             self.assertAllClose(
#                 rbf_kernel.Kdiag(X).eval(), kernel_approx.Kdiag(X).eval(),
#                 rtol=5e-2, atol=5e-2)
#
#
# class TestCosineRBFSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             np.random.seed(123)
#             gamma = np.array([1., 2.])
#             sampler = CosineRBFSampler(2, ls=gamma, n_components=500)
#             kernel_approx = SamplerKernel(sampler)
#             X = np.random.normal(size=[4, 2]).astype(gpf.settings.float_type)
#
#             real_K = np.exp(-np.sum((np.cos(np.expand_dims(X, 1))-np.cos(np.expand_dims(X, 0)))**2  / (2* gamma**2), axis=-1))
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 real_K, kernel_approx.K(X).eval(),
#                 rtol=5e-2, atol=5e-2)
#
#
# class TestLinearSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             sampler = LinearSampler(2)
#             kernel_approx = SamplerKernel(sampler)
#             X = np.random.normal(size=[4, 2]).astype(gpf.settings.float_type)
#             linear_kernel = gpf.kernels.Linear(input_dim=2)
#
#             sess.run(tf.global_variables_initializer())
#             # print(linear_kernel.K(X).eval())
#             # print(kernel_approx.K(X).eval())
#             self.assertAllClose(
#                 linear_kernel.K(X).eval(), kernel_approx.K(X).eval(),
#                 rtol=1e-5, atol=1e-5)
#
#
# class TestCosineSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             gamma = np.array([1., 2.])
#             sampler = CosineSampler(2, ls=gamma)
#             kernel_approx = SamplerKernel(sampler)
#             X = np.random.normal(size=[4, 2]).astype(gpf.settings.float_type)
#             cosine_kernel = gpf.kernels.Cosine(input_dim=2, lengthscales=gamma)
#             sampler._weights = cosine_kernel._weights
#
#             sess.run(tf.global_variables_initializer())
#             a, b = sess.run([cosine_kernel.K(X), kernel_approx.K(X)])
#             self.assertAllClose(a, b, rtol=1e-5, atol=1e-5)
#
#
# class TestCosineV2Sampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             gamma = np.array([1., 0.1])
#             sampler = CosineV2Sampler(2, ls=gamma, n_components=100)
#             kernel_approx = SamplerKernel(sampler)
#             X = np.random.normal(size=[4, 2]).astype(gpf.settings.float_type)
#             cosine_kernel = gpf.kernels.Cosine(input_dim=2, lengthscales=gamma)
#             sampler._weights = cosine_kernel._weights
#
#             sess.run(tf.global_variables_initializer())
#             a, b = sess.run([cosine_kernel.K(X), kernel_approx.K(X)])
#             self.assertAllClose(a, b, rtol=1e-1, atol=1e-1)
#
#
# class TestArcCosineSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             sampler = ArcCosineSampler(2, p=1, n_components=1000)
#             kernel_approx = SamplerKernel(sampler)
#
#             X = np.random.normal(size=[4, 2]).astype(gpf.settings.float_type)
#             arccosine_kernel = gpf.kernels.ArcCosine(input_dim=2, order=1)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 arccosine_kernel.K(X).eval(), kernel_approx.K(X).eval(),
#                 rtol=1e-1, atol=1e-1)
#
#
# class TestConstantSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             sampler = ConstantSampler(2, n_components=1000)
#             kernel_approx = SamplerKernel(sampler)
#
#             X = np.random.normal(size=[4, 2]).astype(gpf.settings.float_type)
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 np.ones([4, 4], dtype=gpf.settings.float_type),
#                 kernel_approx.K(X).eval(),
#                 rtol=1e-1, atol=1e-1)
#
#
# class TestEqApproxSumSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 2.
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=100, scope='sampler1')
#             sampler2 = CosineSampler(input_dim, n_components=100, scope='cosine')
#             sampler3 = LinearSampler(input_dim, n_components=100, scope='linear')
#
#             kernel_sum_approx = SamplerKernel(EqApproxSumSampler(
#                 [sampler1, sampler2, sampler3],
#                 [1., 2., 3.],
#                 n_components=300))
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             kernel3 = SamplerKernel(sampler3)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 (kernel1.K(X)+2*kernel2.K(X)+3*kernel3.K(X)).eval(),
#                 kernel_sum_approx.K(X).eval(), rtol=1e-5, atol=1e-5)
#
# class TestNEqApproxSumSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 2.
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=100, scope='sampler1')
#             sampler2 = CosineSampler(input_dim, n_components=2, scope='cosine')
#             sampler3 = LinearSampler(input_dim, n_components=input_dim, scope='linear')
#
#             sampler_sum_approx = NEqApproxSumSampler(
#                 [sampler1, sampler2, sampler3],
#                 tf.constant([1., 2., 3.]),
#                 n_components=70)
#             kernel_sum_approx = SamplerKernel(sampler_sum_approx)
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             kernel3 = SamplerKernel(sampler3)
#             print(sampler_sum_approx.ns)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 (kernel1.K(X)+2*kernel2.K(X)+3*kernel3.K(X)).eval(),
#                 kernel_sum_approx.K(X).eval(), rtol=5e-2, atol=5e-2)
#
#
#
#
# class TestOuterProductSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 2.
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=2000, scope='sampler1')
#             gamma2 = 1.
#             sampler2 = RBFSampler(input_dim, ls=gamma2, n_components=2000, scope='sampler2')
#
#             kernel_prod_approx = SamplerKernel(OuterProductSampler(sampler1, sampler2))
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             sess.run(tf.global_variables_initializer())
#             a, b = sess.run([kernel1.K(X) * kernel2.K(X), kernel_prod_approx.K(X)])
#             self.assertAllClose(a, b, rtol=1e-5, atol=1e-5)
#
# #
# class TestRandomApproxProdSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 0.5
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=500, scope='sampler1')
#             gamma2 = 1.
#             sampler2 = RBFSampler(input_dim, ls=gamma2, n_components=500, scope='sampler2')
#
#             kernel_prod_approx = SamplerKernel(
#                 RandomApproxProdSampler([sampler1, sampler2], 1000))
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 (kernel1.K(X) * kernel2.K(X)).eval(),
#                 kernel_prod_approx.K(X).eval(), rtol=1e-1, atol=1e-1)
#
#
# class TestOrthogonalApproxProdSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 0.5
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=500, scope='sampler1')
#             gamma2 = 1.
#             sampler2 = RBFSampler(input_dim, ls=gamma2, n_components=500, scope='sampler2')
#
#             kernel_prod_approx = SamplerKernel(
#                 OrthogonalApproxProdSampler([sampler1, sampler2], 400))
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 (kernel1.K(X) * kernel2.K(X)).eval(),
#                 kernel_prod_approx.K(X).eval(), rtol=1e-1, atol=1e-1)
#
#
# class TestSubsetApproxProdSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 0.5
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=2000, scope='sampler1')
#             gamma2 = 1.
#             sampler2 = RBFSampler(input_dim, ls=gamma2, n_components=2000, scope='sampler2')
#
#             kernel_prod_approx = SamplerKernel(
#                 SubsetApproxProdSampler([sampler1, sampler2], 10000))
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 (kernel1.K(X) * kernel2.K(X)).eval(),
#                 kernel_prod_approx.K(X).eval(), rtol=1e-1, atol=1e-1)
#
#
# class TestOneApproxProdSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 4.
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=200, scope='sampler1')
#             gamma2 = 1.
#             sampler2 = RBFSampler(input_dim, ls=gamma2, n_components=200, scope='sampler2')
#
#             kernel_prod_approx = SamplerKernel(
#                 OneApproxProdSampler([sampler1, sampler2], 2000))
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllClose(
#                 (kernel1.K(X) * kernel2.K(X)).eval(),
#                 kernel_prod_approx.K(X).eval(), rtol=1e-1, atol=1e-1)
#
# class TestSketchApproxProdSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 2
#             gamma1 = 4.
#             sampler1 = RBFSampler(input_dim, ls=gamma1, n_components=200, scope='sampler1')
#             gamma2 = 2.
#             sampler2 = RBFSampler(input_dim, ls=gamma2, n_components=200, scope='sampler2')
#
#             kernel_prod_approx = SamplerKernel(
#                 SketchApproxProdSampler([sampler1, sampler2], 1000))
#             kernel1 = SamplerKernel(sampler1)
#             kernel2 = SamplerKernel(sampler2)
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             sess.run(tf.global_variables_initializer())
#             print(sess.run(kernel1.K(X) * kernel2.K(X)))
#             print(kernel_prod_approx.K(X).eval())
#             self.assertAllClose(
#                 (kernel1.K(X) * kernel2.K(X)).eval(),
#                 kernel_prod_approx.K(X).eval(), rtol=1e-1, atol=1e-1)
#
#
# class TestRFFApproxSumSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 3
#             samplers = [RBFSampler(input_dim, ls=2., n_components=100, scope='sampler'+str(i))
#                         for i in range(input_dim)]
#             rbfs = [gpf.kernels.RBF(input_dim, lengthscales=2, name='rbf'+str(i))
#                     for i in range(input_dim)]
#
#             kernel_prod_approx_rff = SamplerKernel(
#                 RFFApproxSumSampler(samplers, 1000))
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             covs_real = tf.add_n([k.K(X) for k in rbfs])
#             sess.run(tf.global_variables_initializer())
#             a, b = sess.run([kernel_prod_approx_rff.K(X), covs_real])
#
#             print(a)
#             print(b)
#             print(np.sum((a-b)**2))
#             self.assertAllClose(a, b, rtol=1e-1, atol=1e-1)
#
#
#
# class TestRFFApproxProdSampler(tf.test.TestCase):
#     def test_kernel(self):
#         with self.test_session() as sess:
#             input_dim = 3
#             samplers = [RBFSampler(input_dim, ls=2., n_components=1000, scope='sampler'+str(i))
#                         for i in range(input_dim)]
#             rbfs = [gpf.kernels.RBF(input_dim, lengthscales=2, name='rbf'+str(i))
#                     for i in range(input_dim)]
#
#             kernels = [SamplerKernel(sampler) for sampler in samplers]
#             kernel_prod_approx_rff = SamplerKernel(
#                 RFFApproxProdSampler(samplers, 1000))
#             kernel_prod_approx_vec = SamplerKernel(
#                 OrthogonalApproxProdSampler([samplers[0], samplers[1], samplers[2]], 1000))
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#
#             covs = tf.reduce_prod([k.K(X) for k in kernels], axis=0)
#             covs_real = tf.reduce_prod([k.K(X) for k in rbfs], axis=0)
#             sess.run(tf.global_variables_initializer())
#             a, b, c, d = sess.run([covs, kernel_prod_approx_rff.K(X), kernel_prod_approx_vec.K(X), covs_real])
#
#             print(a)
#             print(b)
#             print(c)
#             print(np.sum((a-b)**2))
#             print(np.sum((a-c)**2))
#             print(np.sum((d-b)**2))
#             print(np.sum((d-c)**2))
#             self.assertAllClose(a, b, rtol=1e-1, atol=1e-1)
#
#
# ##########################################################################################
#
# class TestListSamplerGroup(tf.test.TestCase):
#     def test(self):
#         with self.test_session() as sess:
#             input_dim = 3
#             sampler1 = RBFSampler(input_dim, n_components=100)
#             sampler2 = LinearSampler(input_dim, n_components=input_dim)
#             sampler3 = CosineSampler(input_dim, n_components=2)
#             group = ListSamplerGroup([sampler1, sampler2, sampler3])
#
#             self.assertAllEqual(len(group), 3)
#             self.assertAllEqual(group.length(0), 100)
#             self.assertAllEqual(group.length(1), input_dim)
#             self.assertAllEqual(group.length(2), 2)
#
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#             feat = group.transform(X)
#             feat1 = sampler1.transform(X)
#             feat2 = sampler2.transform(X)
#             feat3 = sampler3.transform(X)
#
#             sess.run(tf.global_variables_initializer())
#             self.assertAllEqual(feat[0].eval(), feat1.eval())
#             self.assertAllEqual(feat[1].eval(), feat2.eval())
#             self.assertAllEqual(feat[2].eval(), feat3.eval())
#
#
# class TestFCSamplerGroup(tf.test.TestCase):
#     def test(self):
#         with self.test_session() as sess:
#             input_dim = 3
#             sampler1 = RBFSampler(input_dim, n_components=100)
#             sampler2 = LinearSampler(input_dim, n_components=input_dim)
#             sampler3 = CosineSampler(input_dim, n_components=2)
#             list_group = ListSamplerGroup([sampler1, sampler2, sampler3])
#
#             n_samplers, n_components = 4, 70+input_dim+2
#             weights = tf.random_normal(shape=[n_samplers, 3])**2.
#             group = FullyConnectedSamplerGroup(list_group, weights, n_samplers, n_components)
#
#             self.assertAllEqual(len(group), n_samplers)
#             self.assertAllEqual(group.length(0), n_components)
#             self.assertAllEqual(group.length(1), n_components)
#             self.assertAllEqual(group.length(2), n_components)
#             self.assertAllEqual(group.length(3), n_components)
#
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#             feat = group.transform(X)
#             kernel = [tf.matmul(f, f, transpose_b=True) for f in feat]
#
#             feat1 = sampler1.transform(X)
#             k1 = tf.matmul(feat1, feat1, transpose_b=True)
#             feat2 = sampler2.transform(X)
#             k2 = tf.matmul(feat2, feat2, transpose_b=True)
#             feat3 = sampler3.transform(X)
#             k3 = tf.matmul(feat3, feat3, transpose_b=True)
#             feat_bl = [
#                 k1* group.weights[i][0] + k2 * group.weights[i][1] + k3 * group.weights[i][2]
#                 for i in range(n_samplers)
#             ]
#
#             sess.run(tf.global_variables_initializer())
#             aa = sess.run(kernel + feat_bl)
#             self.assertAllClose(aa[0], aa[4], rtol=5e-2, atol=5e-2)
#             self.assertAllClose(aa[1], aa[5], rtol=5e-2, atol=5e-2)
#             self.assertAllClose(aa[2], aa[6], rtol=5e-2, atol=5e-2)
#             self.assertAllClose(aa[3], aa[7], rtol=5e-2, atol=5e-2)
#
#
# class TestRPSamplerGroup(tf.test.TestCase):
#     def test_list_input(self):
#         with self.test_session() as sess:
#             input_dim = 3
#             sampler1 = RBFSampler(input_dim, n_components=100)
#             sampler2 = LinearSampler(input_dim, n_components=input_dim)
#             sampler3 = CosineSampler(input_dim, n_components=2)
#             sampler4 = RBFSampler(input_dim, n_components=50, scope='rbf2')
#             list_group = ListSamplerGroup([sampler1, sampler2, sampler3, sampler4])
#
#             n_components = 1000
#             product_type = 'subset'
#             group = ReduceProdSamplerGroup(list_group, 2, n_components, product_type)
#
#             self.assertAllEqual(len(group), len(list_group) // 2)
#             self.assertAllEqual(group.length(0), n_components)
#             self.assertAllEqual(group.length(1), n_components)
#
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#             feat = group.transform(X)
#             kernel = [tf.matmul(f, f, transpose_b=True) for f in feat]
#
#             feat_fc = list_group.transform(X)
#             kernel_fc = [tf.matmul(f, f, transpose_b=True) for f in feat_fc]
#
#             sess.run(tf.global_variables_initializer())
#             aa = sess.run(kernel + kernel_fc)
#             print(aa[0])
#             print(aa[2]*aa[3])
#             self.assertAllClose(aa[0], aa[2]*aa[3], atol=1e-1, rtol=1e-1)
#             self.assertAllClose(aa[1], aa[4]*aa[5], atol=1e-1, rtol=1e-1)
#
#     def test_fc_input(self):
#         with self.test_session() as sess:
#             input_dim = 3
#             sampler1 = RBFSampler(input_dim, n_components=100)
#             sampler2 = LinearSampler(input_dim, n_components=input_dim)
#             sampler3 = CosineSampler(input_dim, n_components=2)
#             sampler4 = RBFSampler(input_dim, n_components=50, scope='rbf2')
#             list_group = ListSamplerGroup([sampler1, sampler2, sampler3, sampler4])
#
#             fc_group = FullyConnectedSamplerGroup(
#                 list_group, tf.random_uniform([6, 4], 1., 2.), 6, 80)
#
#             n_components = 100000
#             product_type = 'orthogonal'
#             group = ReduceProdSamplerGroup(fc_group, 2, n_components, product_type)
#
#             self.assertAllEqual(len(group), len(fc_group) // 2)
#             self.assertAllEqual(group.length(0), n_components)
#             self.assertAllEqual(group.length(1), n_components)
#
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#             feat = group.transform(X)
#             kernel = [tf.matmul(f, f, transpose_b=True) for f in feat]
#
#             feat_fc = fc_group.transform(X)
#             kernel_fc = [tf.matmul(f, f, transpose_b=True) for f in feat_fc]
#
#             sess.run(tf.global_variables_initializer())
#             aa = sess.run(kernel + kernel_fc)
#             print(aa[0])
#             print(aa[3]*aa[4])
#             self.assertAllClose(aa[0], aa[3]*aa[4], atol=2, rtol=1e-1)
#             self.assertAllClose(aa[1], aa[5]*aa[6], atol=2, rtol=1e-1)
#
# class TestConcatSamplerGroup(tf.test.TestCase):
#     def test(self):
#         with self.test_session() as sess:
#             input_dim = 3
#             sampler1 = RBFSampler(input_dim, n_components=100)
#             sampler2 = LinearSampler(input_dim, n_components=input_dim)
#             sampler3 = CosineSampler(input_dim, n_components=2)
#             list_group = ListSamplerGroup([sampler1, sampler2, sampler3])
#
#             sampler_constant = ConstantSampler(input_dim)
#             group_constant = ListSamplerGroup([sampler_constant])
#             concat_group = ConcatSamplerGroup(list_group, group_constant)
#
#             X = np.random.normal(size=[4, input_dim]).astype(gpf.settings.float_type)
#             feat = concat_group.transform(X)
#
#             feat_fc = list_group.transform(X)
#             feat_con = group_constant.transform(X)
#
#             sess.run(tf.global_variables_initializer())
#             aa = sess.run(feat+feat_fc+feat_con)
#
#             self.assertAllEqual(aa[0], aa[4])
#             self.assertAllEqual(aa[1], aa[5])
#             self.assertAllEqual(aa[2], aa[6])
#             self.assertAllEqual(aa[3], aa[7])
