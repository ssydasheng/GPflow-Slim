# Copyright 2018 Shengyang Sun
# Copyright 2016 James Hensman, Mark van der Wilk,
#                Valentine Svensson, alexggmatthews,
#                PabloLeon, fujiisoup
# Copyright 2017 Artem Artemev @awav
#
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


import enum
import numpy as np
import tensorflow as tf

from gpflowSlim import settings


from gpflowSlim.base import IPrior, ITransform

from gpflowSlim import misc

from gpflowSlim.transforms import Identity


class Parameter(object):
    """
    Parameter class is a cornerstone of the GPflow package. It wraps TensorFlow
    variable and its prior and transformation building operations. In GPflow
    computation graph the parameter is a leaf in the tree.

    *Constrained* and *unconstrained* values of the parameter.

    These definitions arise from optimization topic, constrained and unconstrained
    optimization accordingly.

    _Constrained_ value has acceptable value boundaries and user always works with
    such values. It means that when you pass value to create parameter object, the
    parameter assumes that this is _constrained_ value. When user reads value of
    the parameter, the user gets _constrained_ value.

    For example, variance cannot be negative, hence the constraint is [0, ∞).

    ```
    param = gpflow.Param(1.0)  # Here `1.0` is constrained value.
    value = param.read_value() # `value` is constrained value too.
    ```

    *Unconstrained* value doesn't have any limits. The necessity in this value
    is caused by available set of TensorFlow optimizers. In fact, internal
    parameter tensor is unconstrained and will be trained as unconstrained variable.

    ```
    param = gpflow.Param(1.0)  # `1.0` is constrained value.
    param.parameter_tensor     # unconstrained TensorFlow variable.
    ```

    User can manage contraint type which will be used at a parameter. There is special
    *transform* property and option in constructor for that. The user can pass one of
    the implementations of `ITransform` interface. By default Identity transform is
    applied, so that constrained has no difference from unconstrained, and variable
    is simply considered as unconstrained.

    ```
    param = gpflow.Param(1.0) # Identity transform is applied.
    ```

    In example below, the parameter has exponential transform. It means that input
    value is always must be positive [0, ∞), but internal unconstrained tensor has
    no boundaries (∞, ∞).

    ```
    param = gpflow.Param(1.0, transform=gpflow.transforms.Exp())
    param.read_value()
    # 1.0
    gpflow.get_default_session().run(param.parameter_tensor)
    # -1.0000005000290891e-06
    ```

    *Parameter's shape*.

    The parameter's shape is always fixed by default. It means that user is
    not allowed to modify shape when parameter's tensors are built. User can
    modify default behavior by setting up `fix_shape` option to `False`, the
    parameter will be able to change its shape and internal parameter's tensor
    will have floating shape - None. *NOTE: trainable parameters with floating
    shape cannot be trained by a bunch of TensorFlow optimizers like RMSProp,
    Adam as they require shape information for trainable variables in advance of
    optimizer's tensors construction.*

    :param value: Constrained input value. It can be a float, an integer,
        a float or integer like list, numpy array or TensorFlow variable.
    :param transform: Instance of GPflow.ITransform implementation.
    :param prior: Instance of GPflow.IPrior implementation.
    :param trainable: Boolean flag. It indicates whether variables
        will be added to trainbles TensorFlow set or not.
    :param dtype: Type of new parameter.
    :param fix_shape: Default value is `True` and indicates that shape
        of internal tensor will be the same as the shape of the input
        variable and can not be changed. `False` will turn on floating
        shape mode for the tensor.
    :param name: Name of the parameter.

    :raises: ValueError exception if value is not valid. In cases when
        default graph used for building parameter differs from the graph
        used during construction of priors or transformations, the GPflowError
        exception is raised.
    """

    class ParameterAttribute(enum.Enum):
        PRIOR = 'prior'
        TRANSFORM = 'transform'
        TRAINABLE = 'trainable'

        @property
        def interface(self):
            if self.value == self.PRIOR.value:
                return IPrior
            elif self.value == self.TRANSFORM.value:
                return ITransform

    def __init__(self, value, transform=None, prior=None,
                 trainable=True, dtype=None, name='Param'):
        self.instance_name = name

        if transform is None:
            transform = Identity()
        self.prior = prior
        self.transform = transform
        self.trainable = trainable

        # init var
        vf_value = self.transform.backward(value)
        self.vf_val = tf.get_variable(self.instance_name,
                                      initializer=tf.cast(vf_value, settings.float_type),
                                      trainable=self.trainable)

    @property
    def name(self):
        return self.vf_val.name

    @property
    def shape(self):
        return self.vf_val.shape

    @property
    def dtype(self):
        return self.vf_val.dtype

    @property
    def size(self):
        """The size of this parameter, equivalent to self.value.size"""
        return np.multiply.reduce(self.shape, dtype=np.int32)

    @property
    def value(self):
        return self.transform.forward_tensor(self.vf_val)

    @property
    def unconstrained_tensor(self):
        return self.vf_val

    @property
    def constrained_tensor(self):
        return self.value

    def _build_prior(self, unconstrained_tensor, constrained_tensor):
        """
        Build a tensorflow representation of the prior density.
        The log Jacobian is included.
        """
        if not misc.is_tensor(unconstrained_tensor):
            raise TypeError("Unconstrained input must be a tensor.")

        if not misc.is_tensor(constrained_tensor):
            raise TypeError("Constrained input must be a tensor.")

        prior_name = 'prior'

        if self.prior is None:
            return tf.constant(0.0, settings.float_type, name=prior_name)

        log_jacobian = self.transform.log_jacobian_tensor(unconstrained_tensor)
        logp_var = self.prior.logp(constrained_tensor)
        return tf.squeeze(tf.add(logp_var, log_jacobian, name=prior_name))
