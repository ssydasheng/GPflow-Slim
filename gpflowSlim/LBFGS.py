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

from __future__ import print_function, absolute_import
import tensorflow as tf
import types

# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)

def dot(a, b):
  """Dot product function since TensorFlow doesn't have one."""
  return tf.reduce_sum(a*b)

def verbose_func(s):
  print(s)


def linearize(xs):
    y = []
    for x in xs:
        y.append(tf.reshape(x, [-1]))
    return tf.concat(y, axis=0)


class LBFGS(object):
    """
    :param opfunc: A function return (loss, List([grad, var]))
    """
    def __init__(self, opfunc, max_iter=100, lineSearch=True, lineSearchOptions=None, learningRate=1.,
                 tolFun=3*1e-4, tolX=1e-9, nCorrection=100, verbose=False):
        self.opfunc = opfunc

        #print('linesearch', lineSearch)
        self.maxIter = max_iter
        self.maxEval = self.maxIter * 1.25
        self.tolFun = tolFun
        self.tolX = tolX
        self.nCorrection = nCorrection
        self.lineSearch = lineSearch
        self.lineSearchOpts = lineSearchOptions
        self.learningRate = learningRate
        self.isverbose = verbose

    def get_f_g_x_v(self):
        # evaluate initial f(x) and df/dx
        f, grad_var = self.opfunc()
        grad, vars_ = zip(*grad_var)
        g, x = linearize(grad), linearize(vars_)
        return f, g, x, vars_

    def update_vars(self, vars_, x):
        len_ = 0
        for v in vars_:
            len_v = tf.size(v).numpy()
            v.assign(tf.reshape(x[len_: len_+len_v], tf.shape(v)))
            len_ = len_ + len_v
        assert len_ == tf.shape(x)[0].numpy(), 'Wrong number of variables'
        return

    def eval_f(self, x):
        _, _, old_x, vars_ = self.get_f_g_x_v()
        self.update_vars(vars_, x)
        f, g, _, _ = self.get_f_g_x_v()
        self.update_vars(vars_, old_x)
        return f, g

    def run(self):
        self.history = []

        maxIter = self.maxIter
        maxEval = self.maxEval
        tolFun = self.tolFun
        tolX = self.tolX
        nCorrection = self.nCorrection
        lineSearch = self.lineSearch
        lineSearchOpts = self.lineSearchOpts
        learningRate = self.learningRate
        isverbose = self.isverbose

        # verbose function
        if isverbose:
            verbose = verbose_func
        else:
            verbose = lambda x: None

        # evaluate initial f(x) and df/dx
        f, g, x, vars_ = self.get_f_g_x_v()
        self.history.append((f, g, x, vars_))

        f_hist = [f]
        currentFuncEval = 1
        p = g.shape[0]

        # check optimality of initial point
        tmp1 = tf.abs(g)
        if tf.reduce_sum(tmp1) <= tolFun:
            verbose("optimality condition below tolFun")
            return x, f_hist

        # optimize for a max of maxIter iterations
        nIter = 0
        # times = []
        while nIter < maxIter:

            # keep track of nb of iterations
            nIter = nIter + 1

            ############################################################
            ## compute gradient descent direction
            ############################################################
            if nIter == 1:
                d = -g
                old_dirs = []
                old_stps = []
                Hdiag = 1
            else:
                # do lbfgs update (update memory)
                y = g - g_old
                s = d * t
                ys = dot(y, s)

                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == nCorrection:
                        # shift history by one (limited-memory)
                        del old_dirs[0]
                        del old_stps[0]

                    # store new direction/step
                    old_dirs.append(s)
                    old_stps.append(y)

                    # update scale of initial Hessian approximation
                    Hdiag = ys / dot(y, y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                k = len(old_dirs)

                # need to be accessed element-by-element, so don't re-type tensor:
                ro = [0] * nCorrection
                for i in range(k):
                    ro[i] = 1 / dot(old_stps[i], old_dirs[i])

                # iteration in L-BFGS loop collapsed to use just one buffer
                # need to be accessed element-by-element, so don't re-type tensor:
                al = [0] * nCorrection

                q = -g
                for i in range(k - 1, -1, -1):
                    al[i] = dot(old_dirs[i], q) * ro[i]
                    q = q - al[i] * old_stps[i]

                # multiply by initial Hessian
                r = q * Hdiag
                for i in range(k):
                    be_i = dot(old_stps[i], r) * ro[i]
                    r += (al[i] - be_i) * old_dirs[i]

                d = r
                # final direction is in r/d (same object)

            g_old = g
            f_old = f

            ############################################################
            ## compute step length
            ############################################################
            # directional derivative
            gtd = dot(g, d)

            # check that progress can be made along that direction
            if gtd > -tolX:
                verbose("Can not make progress along direction.")
                break

            # reset initial guess for step size
            if nIter == 1:
                tmp1 = tf.abs(g)
                t = min(1, 1 / tf.reduce_sum(tmp1))
            else:
                t = learningRate

            # optional line search: user function
            lsFuncEval = 0
            if lineSearch:
                # perform line search, using user function
                # f, g, x, t, lsFuncEval = lineSearch(self.opfunc, x, t, d, f, g, gtd, lineSearchOpts)
                # f_hist.append(f)
                alpha = self.strongwolfe(d, x, f, g)
                self.update_vars(vars_, x + alpha * d)
            else:
                # no line search, simply move with fixed-step
                self.update_vars(vars_, x+t*d)

            if nIter != maxIter:
                # re-evaluate function only if not in last iteration
                # the reason we do this: in a stochastic setting,
                # no use to re-evaluate that function here
                f, g, x, vars_ = self.get_f_g_x_v()
                self.history.append((f, g, x, vars_))

                lsFuncEval = 1
                f_hist.append(f)

            # update func eval
            currentFuncEval = currentFuncEval + lsFuncEval

            ############################################################
            ## check conditions
            ############################################################
            if nIter == maxIter:
                break

            if currentFuncEval >= maxEval:
                # max nb of function evals
                verbose('max nb of function evals')
                break

            tmp1 = tf.abs(g)
            # if tf.reduce_sum(tmp1) <= tolFun:#TODO
            if tf.reduce_mean(tmp1) <= tolFun:
                # check optimality
                verbose('optimality condition below tolFun')
                break

            tmp1 = tf.abs(d * t)
            if tf.reduce_sum(tmp1) <= tolX:
                # step size below tolX
                verbose('step size below tolX')
                break

            if tf.abs(f - f_old) < tolX:
                # function value changing less than tolX
                verbose('function value changing less than tolX' + str(tf.abs(f - f_old)))
                break

            if f - f_old > 0.5 * tf.abs(f_old) and nIter > 10:
                diff = tf.abs(self.history[-2][0] - self.history[-3][0])
                abs_ = tf.abs(self.history[-3][0])
                if diff < 0.002 * abs_ or (diff < 0.1 and diff < 0.03 * abs_):
                    self.update_vars(vars_, self.history[-2][2])
                    verbose('Begin to explode, rotating back to previous step')
                    break
                else:
                    raise ValueError('Very unstable, exit')

        return x, f_hist, currentFuncEval

    def Zoom(self, x0, d, alpha_low, alpha_high, fx0, gx0):
        # Algorithms 3.2 on page 59 in
        # Numerical Optimization, by Nocedal and Wright
        # This function is called by strongwolfe
        c1 = 1e-4
        c2 = 0.9
        maxIter = 5

        gx0 = tf.reduce_sum(gx0 * d)
        i = 0
        while True:
            # bisection
            alpha_x = 0.5 * (alpha_low + alpha_high)
            xx = x0 + alpha_x * d
            fxx, gxx = self.eval_f(xx)
            gxx = tf.reduce_sum(gxx * d)

            x_low = x0 + alpha_low * d
            fxl, _ = self.eval_f(x_low)
            if (fxx > fx0 + c1 * alpha_x * gx0) or (fxx >= fxl):
                alpha_high = alpha_x
            else:
                if tf.abs(gxx) <= -c2 * gx0:
                    alpha_star = alpha_x
                    break
                if gxx * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha_x

            i = i + 1
            if i > maxIter:
                alpha_star = alpha_x
                break

        return alpha_star

    def strongwolfe(self, d, x0, fx0, gx0):
        maxIter = 3
        alpha_max, alpha_prev, alpha_x = 20, 0, 1
        c1, c2 = 1e-4, 0.9
        ratio = 0.8

        gx0 = tf.reduce_sum(gx0 * d)
        fx_prev = fx0

        i = 1
        while True:
            xx = x0 + alpha_x * d
            fxx, gxx = self.eval_f(xx)

            gxx = tf.reduce_sum(gxx * d)
            if (fxx > fx0 + c1 * alpha_x * gx0) or ((i > 1) and (fxx >= fx_prev)):
                alpha_star = self.Zoom(x0, d, alpha_prev, alpha_x, fx0, gx0)
                break
            if tf.abs(gxx) <= - c2 * gx0:
                alpha_star = alpha_x
                break
            if gxx >= 0:
                alpha_star = self.Zoom(x0, d, alpha_x, alpha_prev, fx0, gx0)
                break

            alpha_prev, fx_prev = alpha_x, fxx
            if i > maxIter:
                alpha_star = alpha_x
                break

            alpha_x = alpha_x + (alpha_max - alpha_x) * ratio
            i = i + 1

        return alpha_star
