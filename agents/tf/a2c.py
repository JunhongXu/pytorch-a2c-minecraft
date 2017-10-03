import tensorflow as tf
from models import CNNAgent
import gym
import cv2
import numpy as np
from rollouts import Rollouts


class A2C(object):
    """A2C defines loss, optimization ops, and train ops"""
    def __init__(self, env, model, rollout, nstack, grad_clip=1.0, obs_dim=(84, 84, 3),
                 lr=1e-4, entropy_coeff=1e-2, value_coeff=0.5):
        self.sess = tf.Session()
        self.nstack = nstack
        self.model = model(env, self.sess, nstack, obs_dim=obs_dim)
        self.action_holder = tf.placeholder(dtype=tf.int32, shape=(None, ), name='action_holder')
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, ), name='advantage_holder')
        self.returns = tf.placeholder(dtype=tf.float32, shape=(None, ), name='return_holder')

        policy_loss, entropy, value_loss = self.model.loss(self.returns, self.action_holder, self.advantage)
        self.loss = policy_loss - entropy*entropy_coeff + value_coeff*value_loss
        gradients = tf.gradients(self.loss, self.model.trainable_parameters())
        gradients, _ = tf.clip_by_global_norm(clip_norm=grad_clip, t_list=gradients)
        grad_param = [(grad, param) for grad, param in zip(gradients, self.model.trainable_parameters())]

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.apply_gradients(grad_param, name='optimize')
        self.sess.run(tf.global_variables_initializer())

    def step(self, obs, actions, returns, values):
        """Run a training step"""
        advs = returns - values
        loss, _ = self.sess.run([self.loss, self.train_op],
                                feed_dict={self.model.obs: obs, self.action_holder: actions,
                                           self.advantage: advs, self.returns: returns})
        return loss

    def run(self, envs, nupdates, nsteps):
        """store rollouts in side Rollouts object"""
        observations = envs.reset()

        for update in range(nupdates):
            for n in range(nsteps):
                action, sampled_action, value = self.model.act(observations)
                observations, rewards, dones, _ = envs.step(sampled_action)
                rollout.update(n, dones, rewards, sampled_action, action, value)

            _, _, value = self.model.act(observations)
            rollout.values[-1] = value
            rollout.calc_returns(value)

            # update parameters


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    rollout = Rollouts(0.99, 6, 300, env.action_space.n)
    a2c = A2C(env, rollout=rollout, model=CNNAgent, nstack=4)
    obs = cv2.resize(env.reset(), (84, 84))
    obs = np.repeat(obs, 4, axis=2)
    obs = obs[np.newaxis, :, :, :]/255.

    action_logits, sampled_action, values = a2c.model.act(obs)
    print(action_logits, sampled_action, values)

    rewards = np.array([1.0])
    loss = a2c.step(obs, sampled_action, rewards, values)
    print(loss)
