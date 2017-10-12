# Research project on Minecraft tasks.

By using Minecraft game as a testbed, we can test out and compare 
the performance of different existing deep reinforcement learning 
methods in many tasks.

In addition, this project is also going to develop a new deep 
reinforcement learning system that is able to solve a series of tasks
by interacting with human in a sparse rewarded environment.

# Tasks

1. Basic Navigation: really simple environment with only one room. 
Reward: reach goal: +1 else: 0.
2. Normal Navigation: another simple environment, 
but more complex than the first one with two rooms and obstacles.

# TODO
1. Train A2C on the first environment.
2. Implement PPO.
3. Train PPO on the first environment.
4. Test trained A2C on the second environment.
5. Test trained PPO on the second environment.

# Things learned and completed:
1. 09/28/2017: Use multiprocess Process and Pipe to collect data from different processes.
```
    from multiprocess import Process, Pipe
    def collector(worker):
        cmd, data = worker.recv()
        ...
        
    remotes, workers = Pipe()
    p = Process(target=collector, args=(workers, ))
    p.start()
    
    remotes.send(...)
    obs = remotes.recv()
```

2. 10/02/2017: Minimizing cross entropy loss == maximizing log likelihood
    
    In OpenAI A2C baseline implementation, instead of using -log(pi(a)) as the loss function, 
    they use cross entropy: cross_entropy(action_logits, taken_actions) as the policy loss. This
    makes me wonder these two loss functions are equal. It turns out that these two are equal. 
    
3. 10/09/2017: Finished implementing A2C in pytorch and discarded tensorflow version.
    
    OpenAi Gym monitor to record the videos and check learning. 
    
    Some implementation details: 
    
    * when an episode finishes, multiply zeros to that state makes the gradient becomes 0, 
    so this way that observation is not learned by the agent.
    
    * Need to call env.init() before wrapping the environments.
    
    

# Issues
1. Import pytorch and than calling agent.startMission() gives segmentation fault. 
A workaround is using tensorflow for now.

Solved, import pytorch after the first env.reset()
