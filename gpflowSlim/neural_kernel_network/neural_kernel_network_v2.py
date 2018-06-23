from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..kernels import Kernel


class NeuralKernelNetwork(Kernel):
    def __init__(self, input_dim, primitive_kernels, nknWrapper):
        super(NeuralKernelNetwork, self).__init__(input_dim)

        self._primitive_kernels = primitive_kernels
        self._nknWrapper = nknWrapper
        self._parameters = self._parameters + self._primitive_kernels.parameters + self._nknWrapper.parameters

    def Kdiag(self, X, presliced=False):
        primitive_values = [kern.Kdiag(X, presliced) for kern in self._primitive_kernels]
        nkn_outputs = self._nknWrapper.forward(primitive_values)
        return nkn_outputs[0]

    def K(self, X, X2=None, presliced=False):
        primitive_values = [kern.K(X, X2, presliced) for kern in self._primitive_kernels]
        nkn_outputs = self._nknWrapper.forward(primitive_values)
        return nkn_outputs[0]