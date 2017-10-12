from envs.minecraft_wrappers import make_minecraft
from envs.subproc_vec_env import SubprocVecEnv
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--task', default='MinecraftBasic-v0', help='task id')
parser.add_argument('--lr', default=2e-5, help='learning rate')
parser.add_argument('--nprocess', default=6, help='num of processes')
parser.add_argument('--nsteps', default=30, help='num of steps')
parser.add_argument('--e_coeff', default=0.07, help='entropy coefficient')
parser.add_argument('--v_coeff', default=0.5, help='value loss coefficient')
parser.add_argument('--gamma', default=0.99, help='gamma')
parser.add_argument('--nstack', default=4, help='nstack')
parser.add_argument('--testing', action='store_true', default=False)

if __name__ == '__main__':
    args = parser.parse_args()
    nprocess = args.nprocess
    gamma = args.gamma
    nsteps = args.nsteps
    lr = args.lr
    nstack = args.nstack
    e_coeff = args.e_coeff
    v_coeff = args.v_coeff
    res = (40, 40)
    task_id = args.task
    log_dir = 'task_%s/nprocess_%s/gamma_%s/nsteps_%s/lr_%s/nstack_%s/e_coeff_%s/v_coeff%s/res_%s' \
              % (task_id, nprocess, gamma, nsteps, lr, nstack, e_coeff, v_coeff, res)
    is_training = not args.testing
    if is_training:
        if not os.path.exists('checkpoints/'+log_dir):
            os.makedirs('checkpoints/'+log_dir)
        thunk = []
        for i in range(0, nprocess):
            if i == 0:
                _thunk = make_minecraft(task_id, resolution=res, rank=0, seed=i, log_dir='mission_records/'+log_dir, record_fn=lambda x: x % 400 == 0)
            else:
                _thunk = make_minecraft(task_id, resolution=res, rank=0, seed=i)
            thunk.append(_thunk)
        envs = SubprocVecEnv(thunk, minecraft=True)

        from agents.policies import CNNPolicy
        from agents.models import A2C
        import torch

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        a2c = A2C(envs, model=CNNPolicy, nstep=nsteps, nstack=nstack, lr=lr, e_coeff=e_coeff, v_coeff=v_coeff, render=False)
        total = 0
        for e in range(0, 50000):
            episode_obs, episode_rws, episode_values, episode_actions, episode_dones, returns = a2c.run_episode(e)
            loss, policy_loss, mse, advantage, train_values, entropy = a2c.train(returns, episode_obs, episode_actions)
            if e % 1000 == 0:
                torch.save(a2c.model.state_dict(), 'checkpoints/'+log_dir+'/model.pth')
            print(policy_loss.data[0], mse.data[0], entropy.data[0])
        envs.close()
    else:
        env = make_minecraft(task_id=task_id, rank=0, seed=0, resolution=res)()
        from agents.policies import CNNPolicy
        from torch.autograd import Variable
        import torch
        h, w, c = env.observation_space.shape
        c *= nstack
        policy = CNNPolicy((c, h, w), env.action_space.n)
        policy.cuda()
        policy.load_state_dict(torch.load('checkpoints/'+log_dir+'/model.pth'))
        episode_rws = []

        for e in range(0, 200):
            done = False
            obs = env.reset()
            env_total_rws = 0
            while not done:
                env.render()
                obs = np.expand_dims(obs, axis=0)
                obs = np.transpose(obs, (0, 3, 1, 2))
                obs = Variable(torch.from_numpy(obs).float(), volatile=True).cuda()
                action, value = policy.act(obs)
                action, value = action[0, 0], value[0, 0]
                obs, reward, done, _ = env.step(action)
                env_total_rws += reward
            episode_rws.append(env_total_rws)
        print(episode_rws)
