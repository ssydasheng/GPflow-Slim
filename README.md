# GPflow-Slim
GPflow-Slim is a package for building Gaussian process models in python, using TensorFlow. It is adapted from 
[GPflow](https://github.com/GPflow/GPflow) and now contributed by [Shengyang Sun](https://github.com/ssydasheng/Neural-Kernel-Network/) 
and [Guodong Zhang](https://github.com/gd-zhang).

Compared to GPflow, GPflow-Slim enables simpler Tensorflow-style programming. User can define variables arbitrarily
anywhere in the program and apply standard Tensorflow optimizer to optimize the objective.

## Install
After cloning the package, you can install GPflow-Slim by
```
python setup.py install
```
For modifying the source code of GPflow-Slim, it's better to run 
```
python setup.py develop
```

## Examples
Below we show a simple example to use GPflow-Slim and additionally defined variables.
```
X = tf.constant(np.random.normal(size=[20, 4]))
y = tf.sin(X)

var_ = tf.get_variable('var', initializer=1.)
kern = gpf.kernels.RBF(13, ARD=True) + tf.exp(var_)
m = gpf.models.GPR(X, y, kern=kern)

objective = m.objective
optimizer = tf.train.AdamOptimizer(1e-3)
infer = optimizer.minimize(objective)
with tf.Session() as sess:
    sess.run(infer) 
```
For more examples, please refer [examples](./examples) as well as 
[Neural Kernel Network](https://github.com/ssydasheng/Neural-Kernel-Network).


## Citation
To cite this work, please use
```
@article{sun2018differentiable,
  title={Differentiable Compositional Kernel Learning for Gaussian Processes},
  author={Sun, Shengyang and Zhang, Guodong and Wang, Chaoqi and Zeng, Wenyuan and Li, Jiaman and Grosse, Roger},
  journal={arXiv preprint arXiv:1806.04326},
  year={2018}
}
```
as well as 
```
@ARTICLE{GPflow2017,
   author = {Matthews, Alexander G. de G. and {van der Wilk}, Mark and Nickson, Tom and
	Fujii, Keisuke. and {Boukouvalas}, Alexis and {Le{\'o}n-Villagr{\'a}}, Pablo and
	Ghahramani, Zoubin and Hensman, James},
    title = "{{GP}flow: A {G}aussian process library using {T}ensor{F}low}",
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  month = {apr},
  volume  = {18},
  number  = {40},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v18/16-537.html}
}
```

## Ackonwledgement
GPflow-Slim is adapted from [GPflow](https://github.com/GPflow/GPflow).