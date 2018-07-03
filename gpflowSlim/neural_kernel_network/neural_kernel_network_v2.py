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

from ..kernels import Kernel


class NeuralKernelNetwork(Kernel):
    def __init__(self, input_dim, primitive_kernels, nknWrapper):
        super(NeuralKernelNetwork, self).__init__(input_dim)

        self._primitive_kernels = primitive_kernels
        self._nknWrapper = nknWrapper
        self._parameters = self._parameters + self._nknWrapper.parameters
        for kern in self._primitive_kernels:
            self._parameters = self._parameters + kern.parameters

    def Kdiag(self, X, presliced=False):
        primitive_values = [kern.Kdiag(X, presliced) for kern in self._primitive_kernels]
        nkn_outputs = self._nknWrapper.forward(primitive_values)
        return nkn_outputs[0]

    def K(self, X, X2=None, presliced=False):
        primitive_values = [kern.K(X, X2, presliced) for kern in self._primitive_kernels]
        nkn_outputs = self._nknWrapper.forward(primitive_values)
        return nkn_outputs[0]
