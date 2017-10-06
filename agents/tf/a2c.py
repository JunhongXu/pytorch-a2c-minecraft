import tensorflow as tf
from models import CNNAgent
import gym
import cv2
import numpy as np
from envs.atari_wrappers import wrap_deepmind
from envs.env_wrappers import wrapper
from rollouts import Rollouts
from envs.env_wrappers import make_env
from envs.subproc_vec_env import SubprocVecEnv

tf.set_random_seed(0)


class A2C(object):
    """A2C defines loss, optimization ops, and train ops"""
    def __init__(self, env, model, rollout, grad_clip=0.5,
                 lr=1e-4, entropy_coeff=5e-2, value_coeff=0.5):
        self.sess = tf.Session()
        self.env = env
        self.rollout = rollout
        self.model = model(env, self.sess)
        self.action_holder = tf.placeholder(dtype=tf.int32, shape=(None, ), name='action_holder')
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, ), name='advantage_holder')
        self.returns = tf.placeholder(dtype=tf.float32, shape=(None, ), name='return_holder')
        self.h, self.w, self.c = env.observation_space.shape
        self.policy_loss, self.entropy, self.value_loss = self.model.loss(self.returns, self.action_holder, self.advantage)
        self.loss = self.policy_loss - self.entropy*entropy_coeff + value_coeff*self.value_loss
        gradients = tf.gradients(self.loss, self.model.trainable_parameters())
        gradients, _ = tf.clip_by_global_norm(clip_norm=grad_clip, t_list=gradients)
        grad_param = [(grad, param) for grad, param in zip(gradients, self.model.trainable_parameters())]

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
        self.train_op = self.optimizer.apply_gradients(grad_param, name='optimize')
        self.obs = self.env.reset()
        self.rollout.observations[0] = self.obs
        self.sess.run(tf.global_variables_initializer())
        self.rewards = 0
        self.steps = 0

    def step(self, obs, actions, returns, values):
        """Run a training step"""
        advs = returns - values
        loss, policy_loss, value_loss, entropy, _ = self.sess.run([self.loss, self.policy_loss,
                                                                   self.value_loss, self.entropy, self.train_op],
                                                                  feed_dict={self.model.obs: obs,
                                                                             self.action_holder: actions,
                                                                             self.advantage: advs,
                                                                             self.returns: returns})
        return loss, policy_loss, value_loss, entropy

    def colloect_trajectories(self):
        """store rollouts in side Rollouts object"""

        nsteps = self.rollout.nsteps
        for n in range(nsteps):
            # get a_t, v_t
            action, sampled_action, value = self.model.act(self.obs)
            self.env.render()
            # get o_t+1, r_t, done_t
            observations, rewards, dones, _ = self.env.step(sampled_action)
            self.rewards += rewards[0]
            rewards = np.sign(rewards)
            # collect o_t, r_t, a_t, v_t to rollout
            self.rollout.update(n, dones, rewards, action, sampled_action, value, self.obs)
            self.obs = observations
            # get average reward for the first environment
            if dones[0]:
                print('Averaged rewards:%.4f' % (self.rewards))
                self.rewards = 0

        _, _, value = self.model.act(observations)
        self.rollout.values[-1] = value
        self.rollout.calc_returns(value)
        # print(self.rollout.returns)

    def learn(self, nupdates):
        for update in range(nupdates):
            self.rollout.clear()
            self.colloect_trajectories()
            returns = self.rollout.returns[:-1].reshape(-1)
            values = self.rollout.values.reshape(-1)
            actions = self.rollout.actions.reshape(-1)
            observations = self.rollout.observations.reshape(-1, self.h, self.w, self.c)
            _, policy_loss, value_loss, entropy = self.step(observations, actions, returns, values)
            if update % 10 == 0:
                print('Step | Rewards | Policy loss | Value loss | Entropy')
                print('%s | %.4f | %.4f | %.4f | %.4f' %(update, self.rollout.rewards.mean(),
                                                         policy_loss, value_loss, entropy))


if __name__ == '__main__':
    nprocess = 4
    gamma = 0.99
    nsteps = 30

    envs = SubprocVecEnv([make_env('MinecraftBasic-v0', 0, i, '../../logs', wrap=wrapper) for i in range(0, nprocess)], minecraft=True)

    rollout = Rollouts(gamma=gamma, nprocess=nprocess, nsteps=nsteps, nactions=envs.action_space.n,
                        obs_shape=envs.observation_space.shape) # nstack=1)
    a2c = A2C(envs, rollout=rollout, model=CNNAgent)
    a2c.learn(175000)
