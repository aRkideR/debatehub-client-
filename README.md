# pytorch-madrl

This project includes PyTorch implementations of various Deep Reinforcement Learning algorithms for both single agent and multi-agent.

- [ ] A2C
- [ ] ACKTR
- [ ] DQN
- [ ] DDPG
- [ ] PPO

It is written in a modular way to allow for sharing code between different algorithms. In specific, each algorithm is represented as a learning agent with a unified interface including the following components:
- [ ] interact: interact with the environment to collect experience. Taking one step forward and n steps forward are both supported (see `_take_one_step_` and `_take_n_steps`,