---
title:  "Part 4 : Create training environments with OpenAI Gym"
excerpt: "Learning to create virtual environments for training RL agents with the OpenAI Gym library"
toc: true
category:
  - reinforcement learning
---


[Gym](https://www.gymlibrary.ml/) is a Python library maintained by [OpenAI](https://openai.com/). It provides a standard interface to create and initialise training environments for reinforcement learning agents. It also provides a diverse collection of reference environments that represent general RL problems, which can be used to compare the performance of various RL algorithms. In this post, we will learn how to use it.

## Installation

Gym can be installed as a pip package. In a terminal, type:

```bash
  pip install gym
```

## The Cart-Pole environment

The [Cart-Pole environment](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) is a classic RL problem which is provided as a default Gym environment.

This environment simulates an inverted pendulum where a pole is attached to a cart which moves along a frictionless track. At each time step, the agent can move the cart to the left or to the right. The pendulum starts upright, and the goal is to prevent it from falling over. Here's a example.

<iframe width="560" height="315" src="https://www.youtube.com/embed/46wjA6dqxOM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The Cart-Pole problem has been successfully solved in a physical environment, as shown here.

<iframe width="560" height="315" src="https://www.youtube.com/embed/XiigTGKZfks" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/FFW52FuUODQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Initialising the environment

Let's start by creating the environment and by retrieving some useful information.

```python
import gym

env = gym.make('CartPole-v0')

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
```

The action space is discrete of size 2. There are 2 possible actions that can be performed at each time step: move the cart to the left (action: 0) or to the right (action: 1).

The observation space is continuous (Box) of size 4. An observation is an array of 4 numbers that characterise the state of the cart-pole at each time step: [cart position, cart velocity, pole angle, pole angular velocity]. The range values of the states are given below:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

A reward of +1 is given for every time step that the pole remains upright and the cumulative reward is calculated at the end of the episode. The episode terminates if any one of the following occurs:
- Pole Angle is greater than ±12°
- Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
- Episode length is greater than 200

The problem is considered solved when the average reward is greater than or equal to 195 over 100 consecutive trials.  


## Implementing a random policy

The goal of the problem is to identify which series of actions maximise the total cumulative reward at the end of an episode. For the sake of simplicity, we will start by implementing a random policy, i.e. at each time step, the cart is either pushed to the right or to the left randomly.

```python
def policy():
    """ return a random action: either 0 (left) or 1 (right)"""
    action = env.action_space.sample()  
    return action
```

We let the agent learn over 20 episodes of 100 time steps each.

```python
nb_episodes = 20
nb_timesteps = 100

for episode in range(nb_episodes):   # iterate over the episodes
    state = env.reset()              # reset the environment and initialise the state
    rewards = []
    
    for t in range(nb_timesteps):    # iterate over time steps
        env.render()                 # display the environment
        state, reward, done, info = env.step(policy())  # implement the action chosen by the policy
        rewards.append(reward)       # append the reward to the rewards list
        
        if done: # the episode ends either if the pole is &gt; 12 deg from vertical or the cart move by &gt; 2.4 unit from the centre
            cumulative_reward = sum(rewards)
            print("episode {} finished after {} timesteps. Total reward: {}".format(episode, t+1, cumulative_reward))  
            break
    
env.close()
```

Of course, because we are only taking random actions, we can't expect any improvement overtime. The policy is very naive here, we will implement more complex policies later. The code implementing the random policy can be found [here](https://github.com/datamachinist/Random_cartpole). Here is an example of output:

```bash
episode 0 finished after 12 timesteps. Total reward: 12.0
episode 1 finished after 12 timesteps. Total reward: 12.0
episode 2 finished after 16 timesteps. Total reward: 16.0
episode 3 finished after 16 timesteps. Total reward: 16.0
episode 4 finished after 21 timesteps. Total reward: 21.0
episode 5 finished after 23 timesteps. Total reward: 23.0
episode 6 finished after 15 timesteps. Total reward: 15.0
episode 7 finished after 16 timesteps. Total reward: 16.0
episode 8 finished after 19 timesteps. Total reward: 19.0
episode 9 finished after 26 timesteps. Total reward: 26.0
episode 10 finished after 17 timesteps. Total reward: 17.0
episode 11 finished after 15 timesteps. Total reward: 15.0
episode 12 finished after 12 timesteps. Total reward: 12.0
episode 13 finished after 20 timesteps. Total reward: 20.0
episode 14 finished after 17 timesteps. Total reward: 17.0
episode 15 finished after 23 timesteps. Total reward: 23.0
episode 16 finished after 28 timesteps. Total reward: 28.0
episode 17 finished after 12 timesteps. Total reward: 12.0
episode 18 finished after 16 timesteps. Total reward: 16.0
episode 19 finished after 24 timesteps. Total reward: 24.0
```

This video shows the cart pole environment taking random actions for 20 episodes (okay admittedly that's not the most exciting video...).

<iframe width="560" height="315" src="https://www.youtube.com/embed/n3loLvjJAC0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Implementing a hard-coded policy

n the same effort to understand how to use OpenAI Gym, we can define other simple policies to decide what action to take at each time step. For example, instead of using a random policy, we can also hard-code the actions to take at each time steps. For example, we can impose the agent to push the cart to the left for the first 20 time steps and to the right for the other ones.

```python
def policy(t):
   action = 0
   if t &lt; 20:
       action = 0  # go left
   elif t &gt;= 20:
       action = 1  # go right
   return action
```

We can also decide to alternate left and right pushes at each time steps.

```python
def policy(t):
    action = 0
    if t%2 == 1:  # if the time step is odd
        action = 1
    return action
```

This is probably not a very efficient strategy either but you get the idea. 

## Conclusion

In this post, we learnt to initialise a Gym environment and to implement simple policies. In a later article I will explain how to define a more clever policy.