import tensorflow as tf
import numpy as np

from . import settings
from .kronecker import Kronecker

class GridInteprolation(object):
    def __init__(
        self,
        base_kernel,
        grid_size,
        grid_bounds,
        active_dims=None,
    ):
        grid = np.zeros([len(grid_bounds), grid_size], dtype=settings.float_type)
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[i] = tf.linspace(
                grid_bounds[i][0] - grid_diff,
                grid_bounds[i][1] + grid_diff,
                grid_size,
            )

        inducing_points = np.zeros([
            int(pow(grid_size, len(grid_bounds))),
            len(grid_bounds)
        ], dtype=settings.float_type)
        prev_points = None
        for i in range(len(grid_bounds)):
            for j in range(grid_size):
                inducing_points[
                    j * grid_size ** i:(j + 1) * grid_size ** i, i
                ] = grid[i, j]
                if prev_points is not None:
                    inducing_points[
                        j * grid_size ** i:(j + 1) * grid_size ** i, :i
                    ] = prev_points
            prev_points = inducing_points[:grid_size ** (i + 1), :(i + 1)]

        self.inducing_points = tf.constant(inducing_points)
        self.grid = tf.constant(grid)
        self.grid_bounds = grid_bounds
        self.grid_size = grid_size
        self.kernel = base_kernel

    def _inducing_forward(self):
        covs = []
        for id in range(len(self.grid_bounds)):
            cov = self.kernel.Kdim(id, tf.expand_dims(self.grid[id], 1))
            covs.append(cov)
        return Kronecker(covs)

    def _compute_interpolation(self, inputs):
        """
        :param inputs: [n, d]
        :return: sparse interpolation matrix of shape [n, grid_size ** d]
        """
        pass


class Interpolation(object):
    def _cubic_interpolation_kernel(self, scaled_grid_dist):
        """
        Computes the interpolation kernel u() for points X given the scaled
        grid distances:
                                    (X-x_{t})/s
        where s is the distance between neighboring grid points. Note that,
        in this context, the word "kernel" is not used to mean a covariance
        function as in the rest of the package. For more details, see the
        original paper Keys et al., 1989, equation (4).

        scaled_grid_dist should be an n-by-g matrix of distances, where the
        (ij)th element is the distance between the ith data point in X and the
        jth element in the grid.

        Note that, although this method ultimately expects a scaled distance matrix,
        it is only intended to be used on single dimensional data.
        """
        U = tf.abs(scaled_grid_dist)
        res = tf.zeros_like(U, dtype=settings.tf_float)

        U_lt_1 = tf.cast(tf.less(U, 1), dtype=settings.float_type)
        res = res + ((1.5 * U - 2.5) * U**2 + 1) * U_lt_1

        # u(s) = -0.5|s|^3 + 2.5|s|^2 - 4|s| + 2 when 1 < |s| < 2
        U_ge_1_le_2 = 1 - U_lt_1  # U, if U <= 1 <= 2, 0 otherwise
        res = res + (((-0.5 * U + 2.5) * U - 4) * U + 2) * U_ge_1_le_2
        return res

    def interpolate(self, x_grid, x_target, interp_points=range(-2, 2)):

        num_grid_points = tf.shape(x_grid)[1].numpy()
        num_target_points = tf.shape(x_target)[0]
        num_dim = tf.shape(x_grid)[0].numpy()
        num_coefficients = len(interp_points)


        interp_points_flip = tf.cast(interp_points[::-1], settings.tf_float)
        interp_points = tf.cast(interp_points, settings.tf_float)

        interp_values = tf.ones([num_target_points, num_coefficients ** num_dim], dtype=settings.tf_float)
        interp_indices = tf.zeros([num_target_points, num_coefficients ** num_dim], dtype=settings.int_type)

        for i in range(num_dim):
            grid_delta = x_grid[i, 1] - x_grid[i, 0]
            lower_grid_pt_idxs = tf.squeeze(tf.floor((x_target[:, i] - x_grid[i, 0]) / grid_delta))
            lower_pt_rel_dists = (x_target[:, i] - x_grid[i, 0]) / grid_delta - lower_grid_pt_idxs
            lower_grid_pt_idxs = lower_grid_pt_idxs - tf.reduce_max(interp_points)

            scaled_dist = tf.expand_dims(lower_pt_rel_dists, -1) + tf.expand_dims(interp_points_flip, -2)
            dim_interp_values = self._cubic_interpolation_kernel(scaled_dist)

            # Find points who's closest lower grid point is the first grid point
            # This corresponds to a boundary condition that we must fix manually.
            left_boundary_pts = tf.where(lower_grid_pt_idxs < 1)
            num_left = tf.shape(left_boundary_pts)[0].numpy()

            ## only support eager mode for now.
            if num_left > 0:
                left_boundary_pts = tf.squeeze(left_boundary_pts, 1)
                x_grid_first = tf.tile(tf.transpose(tf.expand_dims(x_grid[i, :num_coefficients], 1)), [num_left, 1])
                grid_targets = tf.tile(tf.expand_dims(tf.gather(x_target[:, i], left_boundary_pts), 1), [1, num_coefficients])
                dists = tf.abs(x_grid_first - grid_targets)
                closest_from_first = tf.argmin(dists, 1)

                for j in range(num_left): 
                    dim_interp_values[left_boundary_pts[j], :] = 0
                    dim_interp_values[left_boundary_pts[j], closest_from_first[j]] = 1
                    lower_grid_pt_idxs[left_boundary_pts[j]] = 0

            right_boundary_pts = tf.where(lower_grid_pt_idxs > num_grid_points - num_coefficients)
            num_right = len(right_boundary_pts)

            if num_right > 0:
                right_boundary_pts = tf.squeeze(right_boundary_pts, 1)
                x_grid_last = tf.tile(tf.transpose(tf.expand_dims(x_grid[i, -num_coefficients:], 1)), [num_right, 1])
                grid_targets = tf.tile(tf.expand_dims(tf.gather(x_target[:, i], right_boundary_pts), 1), [1, num_coefficients])
                dists = tf.abs(x_grid_last - grid_targets)
                closest_from_last = tf.argmin(dists, 1)

                for j in range(num_right):
                    dim_interp_values[right_boundary_pts[j], :] = 0
                    dim_interp_values[right_boundary_pts[j], closest_from_last[j]] = 1
                    lower_grid_pt_idxs[right_boundary_pts[j]] = num_grid_points - num_coefficients

            offset = tf.expand_dims(tf.constant(interp_points) - tf.reduce_min(interp_points), -2)
            dim_interp_indices = tf.expand_dims(lower_grid_pt_idxs, -1) + offset

            n_inner_repeat = num_coefficients ** i
            n_outer_repeat = num_coefficients ** (num_dim - i - 1)
            index_coeff = num_grid_points ** (num_dim - i - 1)

            dim_interp_indices = tf.tile(tf.expand_dims(dim_interp_indices, -1), [1, n_inner_repeat, n_outer_repeat])
            dim_interp_values = tf.tile(tf.expand_dims(dim_interp_values, -1), [1, n_inner_repeat, n_outer_repeat])
            interp_indices = interp_indices + tf.reshape(dim_interp_indices, [num_target_points, -1]) * index_coeff
            interp_values = interp_values * tf.reshape(dim_interp_values, [num_target_points, -1])

        return interp_indices, interp_values
