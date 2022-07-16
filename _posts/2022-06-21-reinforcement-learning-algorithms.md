---
title:  "Part 2 : Reinforcement learning algorithms taxonomy"
excerpt: "Reinforcement learning algorithms listed by category"
toc: true
category:
  - reinforcement learning
---



Here is a list of the most common reinforcement learning algorithms grouped by category.

## 1. Model-free RL algorithms

![model free]({{ site.url }}{{ site.baseurl }}/assets/images/model_free.drawio.png)

### 1.1. value-based

- [Q-learning = SARSA max](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) - 1992
- [State Action Reward State-Action (SARSA)](http://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf)  - 1994
- [Deep Q Network (DQN)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) - 2013
    - [Double Deep Q Network (DDQN)](https://arxiv.org/pdf/1509.06461.pdf) - 2015
    - [Deep Recurrent Q Network (DRQN)](https://arxiv.org/abs/1507.06527) - 2015
    - [Dueling Q Network](https://arxiv.org/abs/1511.06581) - 2015
    - [Persistent Advantage Learning (PAL)](https://arxiv.org/abs/1512.04860) - 2015
    - [Bootstrapped Deep Q Network](https://arxiv.org/abs/1602.04621) - 2016
    - [Normalized Advantage Functions (NAF) = Continuous DQN](https://arxiv.org/abs/1603.00748) - 2016
    - [N-Step Q Learning](https://arxiv.org/abs/1602.01783) - 2016
    - [Noisy Deep Q Network (NoisyNet DQN)](https://arxiv.org/abs/1706.10295) - 2017
    - [Deep Q Learning for Demonstration (DqfD)](https://arxiv.org/abs/1704.03732) - 2017
    - [Categorical Deep Q Network = Distributed Deep Q Network = C51](https://arxiv.org/abs/1707.06887) - 2017
    - [Rainbow](https://arxiv.org/abs/1710.02298) - 2017
    - [Quantile Regression Deep Q Network (QR-DQN)](https://arxiv.org/pdf/1710.10044v1.pdf) - 2017
- [Implicit Quantile Network](https://arxiv.org/abs/1806.06923) - 2018
- [Mixed Monte Carlo (MMC)](https://arxiv.org/abs/1703.01310) - 2017
- [Neural Episodic Control (NEC)](https://arxiv.org/abs/1703.01988) - 2017

### 1.2. Policy-based

- [Cross-Entropy Method (CEM)](https://link.springer.com/article/10.1023/A:1010091220143) - 1999
- Policy Gradient
    - [REINFORCE = Vanilla Policy Gradient (VPG)](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) - 1992
    - Policy gradient softmax 
    - [Natural Policy Gradient (Optimisation) (NPG) / (NPO)](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf) - 2002
    - [Truncated Natural Policy Gradient (TNPG)](https://arxiv.org/abs/1604.06778) - 2016

### 1.3. Actor-Critic

- [Advantage Actor Critic (A2C)](https://arxiv.org/abs/1602.01783) - 2016
- [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/abs/1602.01783)  - 2016
- [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) - 2015
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) - 2015
- [Deterministic Policy Gradient (DPG)](http://proceedings.mlr.press/v32/silver14.pdf) - 2014
- [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) - 2015
    - [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/abs/1804.08617) - 2018
    - [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/pdf/1802.09477.pdf) - 2018
- [Actor-Critic with Experience Replay (ACER)](https://arxiv.org/abs/1611.01224) - 2016
- [Actor Critic using Kronecker-Factored Trust Region (ACKTR)](https://arxiv.org/abs/1708.05144) - 2017
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) - 2017
    - [Distributed PPO (DPPO)](https://arxiv.org/abs/1707.02286) - 2017
    - [Clipped PPO (CPPO)](https://arxiv.org/pdf/1707.06347.pdf) - 2017
    - [Decentralized Distributed PPO (DD-PPO)](https://arxiv.org/abs/1911.00357) - 2019
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) - 2018

## 2. Model-based RL algorithms

![model based]({{ site.url }}{{ site.baseurl }}/assets/images/model_based.drawio.png)


### 2.1. Dyna-Style Algorithms / Model-based data generation

- [Dynamic Programming (DP) = DYNA-Q](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.51.7362&rep=rep1&type=pdf) - 1990
- [Embed to Control (E2C)](https://arxiv.org/abs/1506.07365) - 2015
- [Model-Ensemble Trust-Region Policy Optimization (ME-TRPO)](https://arxiv.org/abs/1802.10592) - 2018
- [Stochastic Lower Bound Optimization (SLBO)](https://arxiv.org/abs/1807.03858) - 2018
- [Model-Based Meta-Policy-Optimzation (MB-MPO) (meta learning)](https://arxiv.org/abs/1809.05214) - 2018
- [Stochastic Ensemble Value Expansion (STEVE)](https://arxiv.org/abs/1803.00101) - 2018
- [Model-based Value Expansion (MVE)](https://arxiv.org/abs/1803.00101) - 2018
- [Simulated Policy Learning (SimPLe)](https://arxiv.org/abs/1903.00374) - 2019
- [Model Based Policy Optimization (MBPO)](https://arxiv.org/abs/1906.08253) - 2019

### 2.2. Policy Search with Backpropagation through Time / Analytic gradient computation

- [Differential Dynamic Programming (DDP)](https://www.jstor.org/stable/3613752?origin=crossref&seq=1) - 1970
- [Linear Dynamical Systems and Quadratic Cost (LQR)](http://users.cecs.anu.edu.au/~john/papers/BOOK/B03.PDF) - 1989
- [Iterative Linear Quadratic Regulator (ILQR)](https://homes.cs.washington.edu/~todorov/papers/LiICINCO04.pdf) - 2004
- [Probabilistic Inference for Learning Control (PILCO)](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Deisenroth_ICML_2011.pdf) - 2011
- [Iterative Linear Quadratic-Gaussian (iLQG)](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf) - 2012
- [Approximate iterative LQR with Gaussian Processes (AGP-iLQR)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.716.4271&rep=rep1&type=pdf) - 2014
- [Guided Policy Search (GPS)](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf) - 2013
- [Stochastic Value Gradients (SVG)](https://arxiv.org/abs/1510.09142) - 2015
- [Policy search with Gaussian Process](https://dl.acm.org/doi/10.5555/3306127.3331874) - 2019

### 2.3. Shooting Algorithms / sampling-based planning

- [Random Shooting (RS)](https://arxiv.org/pdf/1708.02596.pdf) - 2017
- [Cross-Entropy Method (CEM)](https://www.sciencedirect.com/science/article/abs/pii/B9780444538598000035) - 2013
    - [Deep Planning Network (DPN)](https://arxiv.org/abs/1811.04551) -2018
    - [Probabilistic Ensembles with Trajectory Sampling (PETS-RS and PETS-CEM)](https://arxiv.org/abs/1805.12114) - 2018
    - [Visual Foresight](https://arxiv.org/abs/1610.00696) - 2016
- [Model Predictive Path Integral (MPPI)](https://arxiv.org/abs/1509.01149) - 2015
    - [Planning with Deep Dynamics Models (PDDM)](https://arxiv.org/abs/1909.11652) - 2019
- [Monte-Carlo Tree Search (MCTS)](https://hal.inria.fr/inria-00116992/document) - 2006
    - [AlphaZero](https://arxiv.org/abs/1712.01815) - 2017

### 2.4. Value-equivalence prediction

- [Value Iteration Network (VIN)](https://arxiv.org/abs/1602.02867) - 2016
- [Value Prediction Network (VPN)](https://arxiv.org/abs/1707.03497) - 2017
- [Model Predictive Control (MPC)](https://arxiv.org/abs/1810.13400) - 2018
- [MuZero](https://arxiv.org/abs/1911.08265) - 2019




## 3. Other RL algorithms


![other RL algo]({{ site.url }}{{ site.baseurl }}/assets/images/other_RL.drawio.png)


### 3.1. General agents

- [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](https://ieeexplore.ieee.org/document/542381) - 1996
- [Episodic Reward-Weighted Regression (ERWR)](https://proceedings.neurips.cc/paper/2008/file/7647966b7343c29048673252e490f736-Paper.pdf) - 2009
- [Relative Entropy Policy Search (REPS)](https://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewFile/1851/2264) - 2010
- [Direct Future Prediction (DFP)](https://arxiv.org/abs/1611.01779) - 2016


### 3.2. Imitation learning

- Behavioral Cloning (BC)
- [Dataset Aggregation (Dagger) (i.e. query the expert)](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) - 2011
- Adversarial Reinforcement Learning
    - [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/abs/1606.03476) - 2016
    - [Adversarial Inverse Reinforcement Learning (AIRL)](https://arxiv.org/abs/1710.11248) - 2017
- [Conditional Imitation Learning](https://arxiv.org/abs/1710.02410) - 2017
- [Soft Q-Imitation Learning (SQIL)](https://arxiv.org/abs/1905.11108) - 2019

### 3.3. Hierarchical RL

- [Hierarchical Actor Critic (HAC)](https://arxiv.org/abs/1712.00948) - 2017

### 3.4. Memory types

- [Prioritized Experience Replay (PER)](https://arxiv.org/abs/1511.05952) - 2015
- [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495) - 2017

### 3.5. Exploration techniques

- E-Greedy
- Boltzmann
- Ornstein–Uhlenbeck process 
- Normal Noise 
- Truncated Normal Noise 
- [Bootstrapped Deep Q Network](https://arxiv.org/abs/1602.04621) 
- [UCB Exploration via Q-Ensembles (UCB)](https://arxiv.org/abs/1706.01502) 
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) 
- [Intrinsic Curiosity Module (ICM) - 2017](https://pathak22.github.io/noreward-rl/)

### 3.6. Meta learning

- [Model-agnostic meta-learning (MAML)](https://arxiv.org/abs/1703.03400) - 2017
- [Improving Generalization in Meta Reinforcement Learning using Learned Objectives (MetaGenRLis)](https://openreview.net/pdf?id=S1evHerYPr) - 2020

### 3.7. Model-free model-based

- [Imagination-Augmented Agents for Deep Reinforcement Learning (I2A)](https://arxiv.org/abs/1707.06203) - 2017
- [Model-Free Model-Based (MB-MF)](https://arxiv.org/abs/1708.02596) - 2017
- [Normalized Adantage Functions (NAF)](https://arxiv.org/abs/1603.00748) - 2018
- [Temporal Difference Models (TDM)](https://arxiv.org/abs/1802.09081) - 2018
