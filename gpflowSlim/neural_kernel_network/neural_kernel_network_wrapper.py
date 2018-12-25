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
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import sympy as sp

from .. import settings
from ..transforms import positive
from ..params import Parameter


class NKNWrapper(object):

    def __init__(self, hparams):
        self._LAYERS = dict(
            Linear=Linear,
            Product=Product,
            Activation=Activation)
        self._build_layers(hparams)

    def _build_layers(self, hparams):
        with tf.variable_scope('NKN'):
            self._layers = [self._LAYERS[l['name']](**l['params']) for l in hparams]

    def forward(self, input):
        with tf.name_scope('NKN'):
            outputs = input # [nm, k]
            for l in self._layers:
                outputs = l.forward(outputs)
        return outputs

    @property
    def parameters(self):
        params = []
        for l in self._layers:
            params = params + l.parameters
        return params

    def symbolic(self):
        ks = sp.symbols(['k'+str(i) for i in range(self._layers[0].input_dim)]) + [1.]
        for l in self._layers:
            ks = l.symbolic(ks)
        assert len(ks) == 1, 'output of NKN must only have one term'
        return ks[0]


class _KernelLayer(object):
    def __init__(self, input_dim, name):
        self.input_dim = input_dim
        self.name = name

    def __call__(self, X):
        assert X.dim == 2, 'Input to KernelLayer must be 2-dimensional'
        with tf.name_scope(self.name):
            self.forward(X)

    def forward(self, input):
        raise NotImplementedError

    @property
    def parameters(self):
       raise NotImplementedError

    def symbolic(self, ks):
        """
        return symbolic formula for the layer
        :param ks: list of symbolic numbers
        :return: list of symbolic numbers
        """
        raise NotImplementedError


class Linear(_KernelLayer):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b` with
    positive weight and bias
    """

    def __init__(self, input_dim, output_dim, name='Linear'):
        super(Linear, self).__init__(input_dim, name=name)
        self.output_dim = output_dim

        with tf.variable_scope(self.name):
            min_w, max_w = 1. / (2 * input_dim), 3. / (2 * input_dim)
            weights = np.random.uniform(low=min_w, high=max_w, size=[output_dim, input_dim]).astype(settings.float_type)
            self._weights = Parameter(weights, transform=positive, name='weights')
            self._bias = Parameter(0.01*np.ones([self.output_dim], dtype=settings.float_type),
                                   transform=positive, name='bias')

    @property
    def weights(self):
        return self._weights.value

    @property
    def bias(self):
        return self._bias.value

    def forward(self, input):
        return tf.matmul(input, tf.transpose(self.weights)) + self.bias

    @property
    def parameters(self):
        return [self._weights, self._bias]

    def symbolic(self, ks):
        out = []
        for i in range(self.output_dim):
            tmp = self.bias.numpy()[i]
            w = self.weights.numpy()
            for j in range(self.input_dim):
                tmp = tmp + ks[j] * w[i, j]
            out.append(tmp)
        return out


class Product(_KernelLayer):
    """
    Applies nodes product.
    """
    def __init__(self, input_dim, step, name='Product'):
        super(Product, self).__init__(input_dim, name=name)
        assert isinstance(step, int) and step > 1, 'step must be number greater than 1'
        assert int(math.fmod(input_dim, step)) == 0, 'input dim must be multiples of step'
        self.step = step

    def forward(self, input):
        output = tf.reshape(input, [tf.shape(input)[0], -1, self.step])
        output = tf.reduce_prod(output, -1)
        return output

    @property
    def parameters(self):
        return []

    def symbolic(self, ks):
        out = []
        for i in range(int(self.input_dim / self.step)):
            out.append(np.prod(ks[i*self.step : (i+1)*self.step]))
        return out


class Activation(_KernelLayer):
    def __init__(self, input_dim, activation_fn, activation_fn_params, name='Activation'):
        super(Activation, self).__init__(input_dim, name=name)
        self.activation_fn = activation_fn
        self.output_dim = input_dim
        self._parameters = activation_fn_params

    def forward(self, input):
        return self.activation_fn(input)

    @property
    def parameters(self):
        return self._parameters

    def symbolic(self, ks):
        return [self.activation_fn(k) for k in ks]
