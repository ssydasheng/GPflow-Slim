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