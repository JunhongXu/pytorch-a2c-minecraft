import gym
from envs.subproc_vec_env import SubprocVecEnv
from envs.env_wrappers import make_env, wrapper, make_minecraft

if __name__ == '__main__':

    # pg = ActorCritic(gym.make('LunarLander-v2'), 0.99, 4e-4)
    # r = 0
    # for i in range(0, 100000):
    #     obs, rws, acts, values, total_reward = pg.run_episode(i)
    #     returns = pg.calculate_returns(rws)
    #     loss, pl, mse, adv, tv, entropy = pg.train(returns[:-1], obs=obs, actions=acts)
    #     r += total_reward
    #     if i % 100 == 0:
    #         print('Update', i, r / 100, mse.data[0], entropy.data[0], pl.data[0])
    #         r = 0
    nprocess = 4
    gamma = 0.99
    nsteps = 30

    envs = SubprocVecEnv([make_minecraft('MinecraftBasic-v0', 0, i) for i in range(0, nprocess)], minecraft=True)
    from agents.pytorch.policies import CNNPolicy, MLP
    from agents.pytorch.models import A2C

    a2c = A2C(envs, model=CNNPolicy, nstep=nsteps, nstack=4, lr=1e-5, e_coeff=0.05)
    total = 0
    for e in range(0, 10000):
        episode_obs, episode_rws, episode_values, episode_actions, episode_dones, returns, final_reward = a2c.run_episode(e)
        total += final_reward
        loss, policy_loss, mse, advantage, train_values, entropy = a2c.train(returns, episode_obs, episode_actions)
        print(policy_loss.data[0], mse.data[0], entropy.data[0])
        print(total/1500)
        total = 0