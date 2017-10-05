import tensorflow as tf
from models import CNNAgent
import gym
import cv2
import numpy as np
from envs.atari_wrappers import wrap_deepmind
from rollouts import Rollouts
from envs.env_wrappers import make_env
from envs.subproc_vec_env import SubprocVecEnv


class A2C(object):
    """A2C defines loss, optimization ops, and train ops"""
    def __init__(self, env, model, rollout, nstack, grad_clip=0.5,
                 lr=7e-4, entropy_coeff=5e-2, value_coeff=0.5):
        self.sess = tf.Session()
        self.nstack = nstack
        self.env = env
        self.rollout = rollout
        self.model = model(env, self.sess)
        self.action_holder = tf.placeholder(dtype=tf.int32, shape=(None, ), name='action_holder')
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, ), name='advantage_holder')
        self.returns = tf.placeholder(dtype=tf.float32, shape=(None, ), name='return_holder')

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

    def step(self, obs, actions, returns, values):
        """Run a training step"""
        advs = returns - values
        loss, policy_loss, value_loss, entropy, _ = self.sess.run([self.loss, self.policy_loss,
                                                                   self.value_loss, self.entropy, self.train_op],
                                                                  feed_dict={self.model.obs: obs,
                                                                             self.action_holder: actions,
                                                                             self.advantage: advs,
                                                                             self.returns: returns})
        print(value_loss, entropy)
        return loss

    def colloect_trajectories(self):
        """store rollouts in side Rollouts object"""

        nsteps = self.rollout.nsteps
        for n in range(nsteps):
            action, sampled_action, value = self.model.act(self.obs)
            self.env.render()
            observations, rewards, dones, _ = self.env.step(sampled_action)
            self.obs = observations
            self.rollout.update(n, dones, rewards, action, sampled_action, value, observations)

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
            observations = self.rollout.observations.reshape(-1, 84, 84, 4)
            self.step(observations, actions, returns, values)
            if update % 10 == 0:
                print('At update step-%s, reward is %s' %(update, self.rollout.rewards.mean()))


if __name__ == '__main__':
    nprocess = 16
    gamma = 0.99
    nsteps = 5

    envs = SubprocVecEnv([make_env('BreakoutNoFrameskip-v4', 0, i, '../../logs', wrap=wrap_deepmind) for i in range(0, nprocess)])
    # envs.init()
    # print(envs, envs.action_space)
    rollout = Rollouts(gamma=gamma, nprocess=nprocess, nsteps=nsteps, nactions=envs.action_space.n,
                        obs_shape=envs.observation_space.shape)
    a2c = A2C(envs, rollout=rollout, model=CNNAgent, nstack=4)
    a2c.learn(5000)
    # a2c.colloect_trajectories()
    # envs.close()
    # print(rollout.observations.shape)
    # print(rollout.rewards.shape)

    # import cv2
    #
    # for p in range(nprocess):
    #     for s in range(nsteps+1):
    #         for stack in range(4):
    #             cv2.imshow('step_%s_process_%s_stack_%s' % (s, p, stack), rollout.observations[s, p, :, :, stack])
    #         cv2.waitKey()
    # obs = cv2.resize(env.reset(), (84, 84))
    # obs = np.repeat(obs, 4, axis=2)
    # obs = obs[np.newaxis, :, :, :]/255.
    #
    # action_logits, sampled_action, values = a2c.model.act(obs)
    # print(action_logits, sampled_action, values)
    #
    # rewards = np.array([1.0])
    # loss = a2c.step(obs, sampled_action, rewards, values)
    # print(loss)
