---
title:  "Part 5 : Q learning for discrete state problems"
excerpt: "Q learning is a simple and efficient way to solve discrete state problems."
category:
  - reinforcement learning
---


In a previous post, we learnt to use the OpenAI Gym library to initialise a training environment and to implement a very simple decision policy (either random or hard-coded). I would recommend that you first go through the first post before reading this one. In this post, we will learn a more sophisticated decision-making algorithm : Q-Learning.

## Q-learning

Q-Learning is a reinforcement learning (RL) algorithm which seeks to find the best action the agent should take given the current state. The goal is to identify a policy that maximises the expected cumulative reward. Q-learning is:
- model-free : it does not build a representation or model of the environment.
- off-policy : the actions taken by the agent are not (always) guided by the policy that is being learnt. i.e. sometimes the agent follows random actions.
- a temporal difference algorithm: the policy is updated after each time step / action taken.
- value-based : it assigns a Q-value for each action being in a given state.

In its most simple form, Q learning uses a **Q-table** that store the Q-values of all state-action pairs possible. It updates this table using the **Bellman equation**. The actions are selected by an **ε-greedy policy**. Let's illustrate these concepts using a simple toy example.

## Q-learning by hand

Let's consider the following problem. The agent (in green) is placed in a 4x4 grid-world environment or maze. Its initial position is in the top-left corner and the goal is to escape the maze by moving to the bottom left corner. The agent has no preliminary knowledge of the environment or the goal to achieve. At each time step, it can perform 4 possible actions: up, down, left or right (unless it is on the edge of the maze). After performing an action, it receives a positive or a negative reward indicated by a score in each cell. There are 16 possible states, shown by a red number. The Q-table is composed of 16 rows (one for each state) and 4 columns (one for each action). For the sake of simplicity, we will initialise the Q-table with zeros, although in practice you'll want to initialise it with random numbers.

![Qlearning0](/assets/images/datamachinist/q_learning/Q00.png)

The agent takes an initial random action, for example 'move right'. 


![Qlearning1](/assets/images/datamachinist/q_learning/Q01.png)

The Q-value associated with being in the initial state (state 1) and moving right is updated using the Bellman equation. In its simplest form, the Bellman equation is written as follows,

![bellman](/assets/images/datamachinist/equations/Selection_003.png)


Where

- <img src="/assets/images/datamachinist/equations/Q.png" alt="Q" width="100"/> is the Q value for being in state s and choosing action a.
- <img src="/assets/images/datamachinist/equations/R.png" alt="R" width="100"/> is the reward obtained from being in state s and performing action a.
- <img src="/assets/images/datamachinist/equations/Selection_005.png" alt="gamma" width="50"/> is the discount factor.
- <img src="/assets/images/datamachinist/equations/Selection_006.png" alt="maxq" width="200"/>  is the maximum Q value of all  possible future actions a' being in the next state s'.

The Q value (Q is for "quality") represents how useful a given action is in gaining some future reward. This equation states that the Q-value yielded from being in a given state and performing a given action is the immediate reward plus the highest Q-value possible from the next state. 

The discount factor controls the contribution of rewards further in the future. Q(s’, a) again depends on Q(s”, a) multiplied by a squared discount factor (so the reward at time step t+2 is attenuated compared to the immediate reward at t+1). Each Q-value depends on Q-values of future states as shown here:

![discount](/assets/images/datamachinist/equations/Selection_007.png)

In our maze problem, if we assume a discount factor equal to 1, the Q value of being in state 0 and performing action 'right' is equal to -1. We can repeat this process by letting the agent take another random action, for example 'down'. The Q-table is updated as follows.

![Qlearning2](/assets/images/datamachinist/q_learning/Q02.png)

Here is a short video to illustrate the whole process.

<iframe width="560" height="315" src="https://www.youtube.com/embed/myXkvoetR8M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Exploration vs exploitation

So far, we let the agent **explore** the environment by taking random actions i.e. it is following a **random policy**. However, after some time exploring, the agent should capitalise on its past experience by selecting the action that is believed to yield the highest expected reward. This is referred to as **exploitation** i.e. the agent follows a **greedy policy**. In order to learn efficiently, there is a trade-off decision to be made between exploration and exploitation. This is implemented in practice by a so-called **ε-greedy** policy. At each time step, a number ε is assigned a value between 0 and 1. Another random number is also selected. If that number is larger than ε, a greedy-action is selected and if it is lower, a random action is chosen. Generally, ε is decreased from 1 to 0 at each time step during the episode. The decay can be linear as shown below but not necessarily.


![decay](/assets/images/datamachinist/exploration-expoloitation.png)
<sub><sup>*[Source](https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/)*</sup></sub>


This process is repeated iteratively as follows.

![Qlearning_process](/assets/images/datamachinist/Qlearning_process.png)
<sub><sup>*[Source](https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/)*</sup></sub>

Eventually, the Q-values should converge to a steady values. The training is completed when the squared loss between the predicted and actual Q value is minimal.


<img src="/assets/images/datamachinist/equations/Selection_008.png" alt="loss" width="400"/> 

Note: a more sophisticated Bellman equation also include the learning rate  as shown below.


![bellman2](/assets/images/datamachinist/equations/Selection_009.png)

If <img src="/assets/images/datamachinist/equations/Selection_010.png" alt="alpha" width="30"/> is equal to 0, the Q value is not updated and if it is equal to 1, we obtain the previous Bellman equation. The learning rate determines to what extent newly acquired information overrides old information.

## Let's see some code -- the taxi problem

Let apply Q learning to a benchmark problem from the OpenAI Gym library: the [Taxi-v2](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py) environment. The taxi problem consists of a 5-by-5 grid world where a taxi can move. The goal is to pick up a passenger at one of the 4 possible locations and to drop him off in another.

### The rules

*There are 4 designated locations on the grid that are indicated by a letter: R(ed), B(lue), G(reen), and Y(ellow). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off, the episode ends. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.*

There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations. (500 = 25*5*4).

There are 6 discrete deterministic actions:
- 0: move down
- 1: move up
- 2: move to the right
- 3: move to the left
- 4: pick up a passenger
- 5: drop-off a passenger

The color coding is as follows:
- blue: passenger
- magenta: destination
- yellow: empty taxi
- green: full taxi
- other letters: locations

### The code

The code can be found [here](https://github.com/PierreExeter/Q-learning-Taxi-V2). 

```python
import numpy as np
import matplotlib.pyplot as plt
import gym
import random

# CREATE THE ENVIRONMENT
env = gym.make("Taxi-v2")
action_size = env.action_space.n
state_size = env.observation_space.n
print("Action space size: ", action_size)
print("State space size: ", state_size)

# INITIALISE Q TABLE TO ZERO
Q = np.zeros((state_size, action_size))

# HYPERPARAMETERS
train_episodes = 2000         # Total train episodes
test_episodes = 100           # Total test episodes
max_steps = 100               # Max steps per episode
alpha = 0.7                   # Learning rate
gamma = 0.618                 # Discounting rate

# EXPLORATION / EXPLOITATION PARAMETERS
epsilon = 1                   # Exploration rate
max_epsilon = 1               # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# TRAINING PHASE
training_rewards = []   # list of rewards

for episode in range(train_episodes):
    state = env.reset()    # Reset the environment
    cumulative_training_rewards = 0
    
    for step in range(max_steps):
        # Choose an action (a) among the possible states (s)
        exp_exp_tradeoff = random.uniform(0, 1)   # choose a random number
        
        # If this number > epsilon, select the action corresponding to the biggest Q value for this state (Exploitation)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(Q[state,:])        
        # Else choose a random action (Exploration)
        else:
            action = env.action_space.sample()
        
        # Perform the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update the Q table using the Bellman equation: Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action]) 
        cumulative_training_rewards += reward  # increment the cumulative reward        
        state = new_state         # Update the state
        
        # If we reach the end of the episode
        if done == True:
            print ("Cumulative reward for episode {}: {}".format(episode, cumulative_training_rewards))
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    # append the episode cumulative reward to the list
    training_rewards.append(cumulative_training_rewards)

print ("Training score over time: " + str(sum(training_rewards)/train_episodes))
```

The cumulative reward vs the number of episode is shown below.


![bellman2](/assets/images/datamachinist/Q-learning-1024x768.png)

Here is the trained taxi agent in action.

<iframe width="560" height="315" src="https://www.youtube.com/embed/fIBy0T3i1oI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Conclusion

Q-learning is one of the simplest Reinforcement Learning algorithms. One of the main limitation of Q learning is that it can only be applied to problems with a finite number of states such as the maze or the taxi example. For environments with a continuous state space - such as the cart-pole problem - it is no longer possible to define a Q table (else it would need an infinite number of rows). It is possible to discretise the state space into buckets in order to solve this limitation, which I will explain in the next article.