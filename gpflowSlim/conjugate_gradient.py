# Copyright 2018 Guodong Zhang
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

import tensorflow as tf


def dot(a, b):
    with tf.name_scope("dot"):
        return tf.reduce_sum(a*b)

def vec(X):
    with tf.name_scope("vec"):
        X = tf.transpose(X)
        return tf.reshape(X, [-1, 1])


def cgsolver(K1, K2, b, C, max_iter=100, tol=1e-6):
    delta = tol * tf.norm(b)
    m = tf.shape(K1)[0]
    n = tf.shape(K2)[0]

    def body(x, k, r_norm_sq, r, p):
        P = tf.transpose(tf.reshape(C * p, [n, m]))
        Ap = C * vec(tf.matmul(tf.matmul(K1, P), K2))
        Ap += p
        alpha = r_norm_sq / dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm_sq_prev = r_norm_sq
        r_norm_sq = dot(r, r)
        beta = r_norm_sq / r_norm_sq_prev
        p = r + beta * p
        return [x, k+1, r_norm_sq, r, p]

    def cond(x, k, r_norm_sq, r, p):
        return tf.logical_and(
            tf.less(delta, r_norm_sq),
            tf.less(k, max_iter))
    r = b
    loop_vars = [
        tf.zeros_like(b), tf.constant(0),
        dot(r, r), r, r]
    return tf.while_loop(
        cond, body, loop_vars)[0]
