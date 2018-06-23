
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
import numpy as np
from sklearn import cluster
from scipy.spatial import distance

import gpflowSlim as gpf


class DataPlaceholders(object):
    def __init__(self):
        self.ximage_flat = tf.placeholder(tf.float32, shape=[None, 28*28])
        self.x_image_reshaped = tf.reshape(self.ximage_flat,[-1, 28, 28, 1], name="img_reshaped")
        self.label = tf.placeholder(tf.int32, shape=[None, 1], name="labels")


def get_mnist():
    mnist = mnist_input_data.read_data_sets(os.path.join(os.path.dirname(__file__), "mnist_data/"), one_hot=False, validation_size=0)

    norm_data = lambda img_in: 2.*img_in - 1.
    add_extra_dim = lambda x: x[:, np.newaxis]

    return (norm_data(mnist.train.images), add_extra_dim(mnist.train.labels),
            norm_data(mnist.validation.images), add_extra_dim(mnist.validation.labels),
            norm_data(mnist.test.images), add_extra_dim(mnist.test.labels))


def create_weight(shape, stddev=0.1, dtype=tf.float32):
    inital = tf.truncated_normal(shape, dtype=dtype, stddev=stddev, name="weight")
    return inital


def create_bias(shape, initial_val=0.1, dtype=tf.float32):
    initial = tf.constant(initial_val, shape=shape, dtype=dtype, name="bias")
    return initial


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name="2dconv")


def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="max_poll2by2")


def make_small_mnist_nn(x_placeholder, end_h=50):
    image_size = 28

    with tf.name_scope("small_convnet"):
        with tf.name_scope("layer1"):
            W1 = tf.get_variable("W_conv1", initializer=create_weight([5,5,1,32]))
            b1 = tf.get_variable("b_1", initializer=create_weight([32]))

            h1 = tf.nn.relu(conv2d(x_placeholder, W1) + b1)
            h1_pooled = max_pool_2by2(h1)

        with tf.name_scope("layer2"):
            num_channels_layer2 = 64
            W2 = tf.get_variable("W_conv2", initializer=create_weight([5, 5, 32, num_channels_layer2]))
            b2 = tf.get_variable("b_2", initializer=create_weight([num_channels_layer2]))

            h2 = tf.nn.relu(conv2d(h1_pooled, W2) + b2)
            h2_pooled = max_pool_2by2(h2)
            layer_length3 = ((image_size/4) ** 2) * num_channels_layer2  # 2 max pools of stride of 2
            h2_pooled_flat = tf.reshape(h2_pooled, [-1, int(layer_length3)])

        with tf.name_scope("layer3"):

            W3 = tf.get_variable("W3", initializer=create_weight([int(layer_length3), 1024]))
            b3 = tf.get_variable("b3", initializer=create_bias([1024]))
            h3 = tf.nn.relu(tf.matmul(h2_pooled_flat, W3) + b3)

        with tf.name_scope("layer4"):
            W4 = tf.get_variable("W4", initializer=create_weight([1024, end_h]))
            b4 = tf.get_variable("b4", initializer=create_bias([end_h]))
            h4 = tf.matmul(h3, W4) + b4

    return h4


def suggest_good_intial_inducing_points(phs: DataPlaceholders, x_data, h, tf_session, num_inducing):
    h_data = tf_session.run(h, feed_dict={phs.ximage_flat: x_data})
    kmeans = cluster.MiniBatchKMeans(n_clusters=num_inducing, batch_size=num_inducing*10)
    kmeans.fit(h_data)
    new_inducing = kmeans.cluster_centers_
    return new_inducing


def suggest_sensible_lengthscale(phs: DataPlaceholders, x_data, h, tf_session):
    h_data = tf_session.run(h, feed_dict={phs.ximage_flat: x_data})
    lengthscale = np.mean(distance.pdist(h_data, 'euclidean'))
    return lengthscale


def main():
    """
    Simple demonstration of how you can put a GP on top of a NN and train the whole system end-to-end in GPflow-1.0.


    Note
    that in the new GPflow there are new features that we do not take advantage of here but could be used to make
    the whole example cleaner. For example you may want to use a gpflow.train Optimiser as this will take care of
    passing in the GP model feed dict for you as well as initially initialising the optimisers TF variables.
    You could also choose to tell the gpmodel to initialise the NN variables by subclassing SVGP and overriding the
    appropriate variable initialisation method.
    """
    # ## We load in the MNIST data. We will create a validation set but will not use it in this simple example.
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist()
    rng = np.random.RandomState(100)
    train_permute = rng.permutation(x_train.shape[0])
    x_train, y_train = x_train[train_permute, :], y_train[train_permute, :]

    # ## We set up a TensorFlow Graph and a Session linked to this.
    # tf_graph = tf.get_default_graph()
    tf_session = tf.Session()

    # ## We have some settings for the model and its training which we will set up below.
    num_h = 100
    num_classes = 10
    num_inducing = 100
    minibatch_size = 250

    # ## We set up the NN part of the GP kernel. This needs to be put on the same graph
    # with tf_graph.as_default():
    phs = DataPlaceholders()
    nn_base = tf.make_template("sconvnet_kernel", make_small_mnist_nn, end_h=num_h)  # end h is the number of hidden
    # units at the end

    h = nn_base(phs.x_image_reshaped)
    h = tf.cast(h, gpf.settings.tf_float)

    nn_vars = tf.global_variables()  # only nn variables exist up to now.
    tf_session.run(tf.variables_initializer(nn_vars))

    # ## We now set up the GP part. Instead of the usual X data it will get the data after being processed by the NN.
    kernel = gpf.kernels.RBF(num_h, lengthscales=suggest_sensible_lengthscale(phs, x_train[:5000, :], h, tf_session), ARD=True)
    likelihood = gpf.likelihoods.MultiClass(num_classes)
    gp_model = gpf.models.SVGP(h, phs.label, kernel, likelihood,
                               Z=suggest_good_intial_inducing_points(phs, x_train[:5000, :], h, tf_session, num_inducing),
                           num_latent=num_classes, whiten=False, minibatch_size=None, num_data=x_train.shape[0])
    # ^ so we say minibatch size is None to make sure we get DataHolder rather than minibatch data holder, which
    # does not allow us to give in tensors. But we will handle all our minibatching outside.
    # gp_model.compile(tf_session)

    # ## We create ops to measure the predictive log likelihood and the accuracy.
    # with tf_graph.as_default():
    log_likelihood_predict = gp_model.likelihood.predict_density(*gp_model._build_predict(h), phs.label)
    accuracy = tf.cast(tf.equal(tf.argmax(gp_model.likelihood.predict_mean_and_var(*gp_model._build_predict(h))[0], axis=1, output_type=tf.int32),
                                     tf.squeeze(phs.label)), tf.float32)
    avg_acc = tf.reduce_mean(accuracy)
    avg_ll = tf.reduce_mean(log_likelihood_predict)

    optimiser = tf.train.AdamOptimizer()
    objective = gp_model.objective
    minimise = optimiser.minimize(objective)
    other_vars = list(set(tf.global_variables()) - set(nn_vars))
    tf_session.run(tf.variables_initializer(other_vars))
    # print(tf.global_variables())

    # ## We now go through a training loop where we optimise the NN and GP. we will print out the test results at
    # regular intervals.
    run_metadata = tf.RunMetadata()

    data_indx = 0
    print("starting")
    for i in range(6000):
        indx_array = np.mod(np.arange(data_indx, data_indx + minibatch_size), x_train.shape[0])
        data_indx += minibatch_size

        fd = {phs.ximage_flat: x_train[indx_array], phs.label: y_train[indx_array]}
        _, loss_evd = tf_session.run(
            [minimise, -objective], feed_dict=fd,
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
        # Print progress every 50 steps.
        if i % 50 == 0:
            fd = {phs.ximage_flat: x_test, phs.label: y_test}
            accuracy_evald, log_like_evald = tf_session.run([avg_acc, avg_ll], feed_dict=fd)
            print("Iteration {}: Loss is {}. \nTest set LL {}, Acc {}".format(i, loss_evd, log_like_evald,
                                                                              accuracy_evald))

    from tensorflow.python.client import timeline
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    trace_file = open('timeline.ctf.json', 'w')
    trace_file.write(trace.generate_chrome_trace_format())

if __name__ == '__main__':
    main()
