from envs.minecraft_wrappers import make_minecraft
from envs.subproc_vec_env import SubprocVecEnv

if __name__ == '__main__':
    nprocess = 5
    gamma = 0.99
    nsteps = 30
    lr = 1e-5
    nstack = 4
    e_coeff = 0.02
    v_coeff = 0.5
    task_id = 'MinecraftBasic-v0'
    log_dir = 'mission_records/task_%s/nprocess_%s/gamma_%s/nsteps_%s/lr_%s/nstack_%s/e_coeff_%s/v_coeff%s' \
              % (task_id, nprocess, gamma, nsteps, lr, nstack, e_coeff, v_coeff)

    thunk = []
    for i in range(0, nprocess):
        if i == 0:
            _thunk = make_minecraft(task_id, rank=0, seed=i, log_dir=log_dir, record_fn=lambda x: x % 1000 == 0)
        else:
            _thunk = make_minecraft(task_id, rank=0, seed=i)
        thunk.append(_thunk)
    envs = SubprocVecEnv(thunk, minecraft=True)

    from agents.policies import CNNPolicy
    from agents.models import A2C

    a2c = A2C(envs, model=CNNPolicy, nstep=nsteps, nstack=nstack, lr=lr, e_coeff=e_coeff, v_coeff=v_coeff)
    total = 0
    for e in range(0, 50000):
        episode_obs, episode_rws, episode_values, episode_actions, episode_dones, returns, final_reward = a2c.run_episode(e)
        total += final_reward
        loss, policy_loss, mse, advantage, train_values, entropy = a2c.train(returns, episode_obs, episode_actions)
        print(policy_loss.data[0], mse.data[0], entropy.data[0])
        print(total/1500)
        total = 0
