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

import gpytorch.utils.interpolation as pytorch_Interpolation
import torch
from torch.autograd import Variable
import numpy as np

import tensorflow as tf
from .interpolation_np import GridInteprolation, Interpolation
from . import settings


class TestInterpolation(tf.test.TestCase):
    def test(self):
        with self.test_session() as sess:
            interpolation_class = GridInteprolation(None, grid_size=10, grid_bounds=[(0, 1), (0, 1)])
            x_targets = np.random.uniform(size=[13, 2]).astype(settings.float_type)

            tf_inter = Interpolation().interpolate(interpolation_class.grid, x_targets)
            pt_inter = pytorch_Interpolation.Interpolation().interpolate(
                Variable(torch.from_numpy(interpolation_class.grid)),
                Variable(torch.from_numpy(x_targets)))

            print(tf_inter[0])
            print(pt_inter[0].data.numpy())

            print(tf_inter[1])
            print(pt_inter[1].data.numpy())

            self.assertAllClose(tf_inter[0], pt_inter[0].data.numpy())
            self.assertAllClose(tf_inter[1], pt_inter[1].data.numpy())