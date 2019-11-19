[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

#### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym. 

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 


5. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

6. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

#### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

# Report

In order to solve the environment, our agent must achieve a score of +30 averaged across all 20 agents for 100 consecutive episodes.

## Global architecture

- First, I needed to determine how many "brains" were controlling the agents.

The version of the environment I've chosen has 20 different arms, whereas the Navigation project had only a single agent. To keep things simple, I decided to use a single brain to control all 20 agents for temporal and technical simplicity.

- I also needed to pick which algorithms were the most suitable for the Reacher environment.

The action space is continuous, which allows each agent to execute more complex actions, whilst the agent in the first project was limited to 4 discrete actions.
Given this parameter, the value-based (Deep Q-Network) method I used for the last project would not work as well. That's why I needed to use policy-based methods.

## DDPG

In [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf), written by researchers at Google DeepMind, they highlight that Deep Deterministic Policy Gradient (DDPG) can be seen as an extension of Deep Q-learning to continuous tasks. That's why I picked this "Actor-Critic" algorithm.

With a policy-based approach, the Actor learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent, while the Critic learns how to estimate the value of different state-action pairs with a value-based approach.

## Local and target networks 

Moreover, I used local and target networks to improve stability.

## Ornstein-Uhlenbeck process

I've also used the Ornstein-Uhlenbeck process (as suggested in the previously mentioned paper by Google DeepMind), which adds a certain amount of noise to the action values at each timestep and therefore allows the arm to maintain velocity and explore the action space.

## Gradient clipping

Finally, after many failed trainings, I've decided to use gradient clipping, implemented in `Agent.learn()`, within `ddpg_agent.py`. This method sets a limit on the size of the parameter updates, and stops them from growing too fast.

# Hyperparameters

## Networks

In `model.py`, you can find the (almost similar) architectures of the Actor and the Critic :
- input_size = state_size = 33
- 2 hidden fully-connected layers with 400 and 300 nodes
- ReLu activation function was used between fc1 and fc2
- A Batch Normalization was used between the output of fc1 and its activation

## Agent's hyperparameters

Many tests were run but the final choice of the hyperparameters was :

- `BUFFER_SIZE` = int(1e6) : replay buffer size
- `BATCH_SIZE` = 128 : minibatch size
- `GAMMA` = 0.99 : discount factor
- `TAU` = 1e-3 : for soft update of target parameters
- `LR` = 5e-4 : learning rate
- `LR_ACTOR` = 1e-3         # learning rate of the actor
- `LR_CRITIC` = 1e-3        # learning rate of the critic
- `WEIGHT_DECAY` = 0        # L2 weight decay
- `EPSILON = 1.0`           # explore->exploit for Ornstein-Uhlenbeck process
- `EPSILON_DECAY = 1e-6`    # decay rate for noise process

# Results

The agent reached its goal (moving average >= 30 over 100 consecutive episodes) after 168 episodes.
The training details are shown below.

<img src="assets/training_scores.png" width="100%" align="center" alt="" title="Training Scores" />

<img src="assets/training_plot.png" width="100%" align="center" alt="" title="Plot" />

# Future work

Testing another algorithm such as Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), or Distributed Distributional Deterministic Policy Gradients (D4PG) would be more efficient.
