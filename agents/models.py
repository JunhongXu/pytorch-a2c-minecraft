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
        return action, value.reshape(-1)

    def loss(self, rs, actions, advantage):
        """
        Calculates the components of policy gradient: log\pi, advantage function, and entropy.
        These are used in the training step.

        rs: placeholder, holds the value of returns, shape(nenvs*nsteps)
        actions: placeholder, holds the value of actions taken in each trail shape(nenvs*nsteps)
        advantage: placeholder, holds the value of advantage estimation shape(nenvs*nsteps)
        """
        # negative log probability loss -log(softmax(actions))
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy, labels=actions, name='nll')
        policy_loss = tf.reduce_mean(neg_log_prob * advantage, name='policy_loss')

        # nenvs*nsteps, nactions
        action_probs = tf.nn.softmax(self.policy)
        entropy = -tf.nn.log_softmax(action_probs) * action_probs
        entropy = tf.reduce_mean(tf.reduce_sum(entropy, axis=1), name='entropy')

        value_loss = tf.reduce_mean(tf.square(self.value - rs), name='value_loss')
        return policy_loss, entropy, value_loss


if __name__ == '__main__':
    import gym
    tf.set_random_seed(0)
    np.random.seed(0)
    env = gym.make('Breakout-v0')
    sess = tf.InteractiveSession()
    cnn = CNNAgent(env, sess, 4)
    action_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, ))
    advantage = tf.placeholder(dtype=tf.float32, shape=(None, ))
    returns = tf.placeholder(dtype=tf.float32, shape=(None, ))
    print(env.observation_space.shape)
    obs = np.random.randn(1, 210, 160, 3*4)


    actions, values = cnn.act(obs)
    rs = np.array([1])
    adv = rs - values

    print('action logits', actions)
    actions = np.argmax(actions, axis=1)
    print('taken actions', actions, 'estimated value', values, 'advantage', adv)
    policy_loss, entropy, value_loss = cnn.loss(returns, action_placeholder, adv)

    p, e, v = sess.run([policy_loss, entropy, value_loss], {returns: rs, action_placeholder: actions, advantage: adv, cnn.obs: obs})
    print(p, e, v)