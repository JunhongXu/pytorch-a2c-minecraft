import tensorflow as tf
from tensorflow.contrib.layers import *
import numpy as np


def conv(x, out_channel, kernel_size, stride, name, init_scale=np.sqrt(2), padding='SAME'):
    with tf.variable_scope(name):
        x = conv2d(
            x, num_outputs=out_channel, kernel_size=kernel_size,
            padding=padding, stride=stride, activation_fn=tf.nn.elu,
            weights_initializer=tf.orthogonal_initializer(gain=init_scale)
        )
    return x


def dense(x, num_outputs, name, initializer, activation_fn=tf.nn.elu):
    with tf.variable_scope(name):
        shape = x.get_shape().ndims
        if shape > 2:
            x = flatten(x)

        x = linear(x, num_outputs, weights_initializer=initializer, activation_fn=activation_fn)
    return x


class CNNAgent(object):
    def __init__(self, env, sess, nstack, beta=0.1, reuse=False):
        """
                                                            -> (action, action_space)
            Construct a CNN agent. 3->64->128->256->(fc, 512)
                                                            -> (value, 1)
        """
        h, w, c = env.observation_space.shape
        action_space = env.action_space.n
        self.sess = sess
        self.obs_shape = (None, h, w, c*nstack)
        self.obs = tf.placeholder(tf.float32, self.obs_shape, name='observations')
        self.beta = beta

        with tf.variable_scope('model', reuse=reuse):
            conv1 = conv(self.obs, 32, 8, 4, 'conv1')
            conv2 = conv(conv1, 64, 4, 2, 'conv2')
            conv3 = conv(conv2, 64, 3, 1, 'conv3')
            fc1 = dense(conv3, 256, 'fc1', initializer=tf.orthogonal_initializer(1))
            self.policy = dense(fc1, action_space, 'policy', initializer=tf.orthogonal_initializer(1), activation_fn=None)
            self.value = dense(fc1, 1, 'value', initializer=tf.orthogonal_initializer(1), activation_fn=None)

        self.sess.run(tf.global_variables_initializer())

    def act(self, obs):
        """
        obs is of shape (nenvs*nsteps, h, w, c*nstacks)
        """
        action, value = self.sess.run([self.policy, self.value], feed_dict={self.obs: obs})
        return action, value

    def policy_gradient(self, obs, rs, advantage):
        """
        Calculates the components of policy gradient: log\pi, advantage function, and entropy.
        These are used in the training step.

        obs: numpy array, observations from environments
        rs: placeholder, holds the value of returns
        """
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(self.policy)
        advantage = rs - self.value
        entropy = self.beta *


if __name__ == '__main__':
    import gym
    env = gym.make('Breakout-v0')
    sess = tf.InteractiveSession()
    cnn = CNNAgent(env, sess, 4)
    print(env.observation_space.shape)
    obs = np.random.randn(100, 210, 160, 3*4)
    print(cnn.act(obs))
