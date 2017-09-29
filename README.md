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
1. Implement A2C.
2. Train A2C on the first environment.
3. Implement PPO.
4. Train PPO on the first environment.
5. Test trained A2C on the second environment.
6. Test trained PPO on the second environment.