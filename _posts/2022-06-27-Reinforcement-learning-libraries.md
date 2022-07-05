---
title:  "Part 9 : Reinforcement learning libraries"
excerpt: "Let's compare some reinforcement learning libraries"
toc: true
category:
  - reinforcement learning
---

In previous posts, we sometimes implemented manually some RL algorithms to explain how they work. However in practice, we will use some optimised libraries that implement common RL algorithms instead. In this post, we will compare some of these libraries. A list of such libraries can be found [here](https://github.com/godmoves/reinforcement_learning_collections).



## Main characteristics

| RL Libraries     | Framework                    | Tensorboard support     | Custom environment interface |
|------------------|------------------------------|-------------------------|------------------------------|
| Keras-RL         | Keras                        | No                      | No                           |
| Tensorforce      | Tensorflow                   | Yes                     | Yes                          |
| OpenAI Baselines | Tensorflow                   | ?                       | No                           |
| Stable baselines | Tensorflow                   | Yes                     | Yes                          |
| TF Agents        | Tensorflow                   | Yes                     | ?                            |
| Ray / Rllib      | Tensorflow / Pytorch / Keras | Yes                     | Yes                          |
| Tensorlayer      | Tensorflow                   | Yes                     | ?                            |
| Rllab / Garage   | Tensorflow / Pytorch         | ?                       | Yes                          |
| Coach            | TensorFlow                   | No but custom dashboard | Yes                          |


## Algorithms implemented

| RL Libraries     | DQN | DDPG | NAF / CDQN | CEM | SARSA | DqfD | PG / REINFORCE | PPO | A2C | A3C | TRPO | GAE | ACER | ACKTR | GAIL | SAC | TD3 | ERWR | NPO | REPS | TNPG | CMA-ES | MMC | PAL | TDM | RIG | Skew-Fit |
|------------------|-----|------|------------|-----|-------|------|----------------|-----|-----|-----|------|-----|------|-------|------|-----|-----|------|-----|------|------|--------|-----|-----|-----|-----|----------|
| Keras-RL         | X   | X    | X          | X   | X     |      |                |     |     |     |      |     |      |       |      |     |     |      |     |      |      |        |     |     |     |     |          |
| Tensorforce      | X   | X    | X          |     |       | X    | X              | X   | X   | X   | X    | X   |      |       |      |     |     |      |     |      |      |        |     |     |     |     |          |
| OpenAI Baselines | X   | X    |            |     |       |      |                | X   | X   |     | X    |     | X    | X     | X    |     |     |      |     |      |      |        |     |     |     |     |          |
| Stable baselines | X   | X    |            |     |       |      |                | X   | X   |     | X    |     | X    | X     | X    | X   | X   |      |     |      |      |        |     |     |     |     |          |
| TF Agents        | X   | X    |            |     |       |      | X              | X   |     |     |      |     |      |       |      | X   | X   |      |     |      |      |        |     |     |     |     |          |
| Ray / Rllib      | X   | X    |            |     |       |      | X              | X   | X   | X   |      |     |      |       |      | X   | X   |      |     |      |      |        |     |     |     |     |          |
| Tensorlayer      | X   | X    |            |     |       |      | X              | X   | X   | X   | X    |     |      |       |      | X   | X   |      |     |      |      |        |     |     |     |     |          |
| Rllab / Garage   | X   | X    |            | X   |       |      | X              | X   |     |     | X    |     |      |       |      |     | X   | X    | X   | X    | X    | X      |     |     |     |     |          |
| Coach            | X   | X    | X          |     |       |      | X              | X   |     | X   |      | X   | X    |       |      | X   | X   |      |     |      |      |        | X   | X   |     |     |          |
| Rlkit            | X   |      |            |     |       |      |                |     |     |     |      |     |      |       |      | X   | X   |      |     |      |      |        |     |     | X   | X   | X        |