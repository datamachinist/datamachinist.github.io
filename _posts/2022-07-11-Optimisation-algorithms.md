---
title:  "Part 1 - Optimisation algorithms"
excerpt: "A non-exhaustive list of optimisation algorithms, classed by categories."
toc: true
category:
  - optimisation
---


In this post, we will list some important optimisation algorithms, classed by categories.


**Legend:**
- M = Multi-objective
- C = Constrained
- U = Unconstrained
- I = Integer programming
- sto = Stochastic

## 1. Global optimisation


![Optimisation_algo_global]({{ site.url }}{{ site.baseurl }}/assets/images/Optimisation_algo_global.png)

### 1.1. Evolutionary optimisation - Population-based


#### 1.1.1. Genetic algorithms

- Simple Genetic Algorithm (GA) - S / U / I / Sto
- Non-dominated Sorting Genetic Algorithm (NSGA-II) - M / U / I - 2000
- Epsilon-Non-dominated Sorting Genetic Algorithm (ε-NSGA-II) - M / U / I - 2006
- Non-dominated Sorting Genetic Algorithm (NSGA-III) - 2014
- Reference point based Non-dominated Sorting Genetic Algorithm (R-NSGA-II) - 2018
- Unified Non-dominated Sorting Genetic Algorithm (U-NSGA-III) - 2016
- Vector Evaluated Genetic Algorithms (VEGA) - 1985
- Adapting Scatter Search to Multiobjective Optimization (AbYSS) - 2008
- Fast Pareto Genetic Algorithm (FastPGA) - 2007
- Multi-Objective Cellular Genetic Algorithm (MOCell) - 2006
- Multi-Objective Cross generational elitist selection, Heterogeneous recombination, Cataclysmic mutation (MOCHC) - 2007
- Biased Random Key Genetic Algorithm (BRKGA) 
- Multi-objective Quantum-inspired Evolutionary Algorithm (MQEA)
- Reference Point-Based Nondominated Sorting Multi-Objective Quantum-Inspired Evolutionary Algorithm (RN-MQEA)


#### 1.1.2. Differential Evolution

- Differential Evolution (DE) - S / U
- Self-adaptive Differential Evolution (jDE, iDE and pDE) - S / U
- Generalized Differential Evolution 3 (GDE3) - 2005
- Cellular Differential Evolution (CellDE) 


#### 1.1.3. Evolution Strategy

- Evolution Strategy (ES) - S
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES) - M / S / U / Sto - 2007
- Multi-Objective Covariance Matrix Adaptation Evolution Strategy (MO-CMA-ES)
- Bi-Population Covariance Matrix Adaptation Evolution Strategy (BI-POP CMA-ES)
- Exponential Evolution Strategy (xNES) - S / U / Sto
- Pareto Archived Evolutionary Strategy (PAES) - 1999 


#### 1.1.4. Indicator-based

- Indicator Based Evolutionary Algorithm (IBEA) - 2004
- Simple Indicator-Based Evolutionary Algorithm (SIBEA)
- Sampling-Based Hypervolume-Oriented Algorithm (SHV)
- Hypervolume Estimation Algorithm for Multiobjective Optimization (HypE)
- S-Metric Selection Evolutionary Multiobjective Optimisation Algorithm (SMS-EMOA) - 2007
- Set Preference Algorithm for Multiobjective Optimization (SPAM) 


#### 1.1.5. Decomposition algorithms

- Multi-Objective Evolutionary Algorithm with Decomposition (MOEA/D) - M / U - 2009
- Decomposition-Based Evolutionary Algorithm (DBEA) - 2015 


#### 1.1.6. Other evolutionary algorithms

- (N+1)-ES Simple Evolutionary Algorithm (SEA) - S / U / Sto
- Epsilon-Multi-Objective Evolutionary Algorithm (ε-MOEA) - 2003
- Pareto Envelope-Based Selection Algorithm II (PESA-II) - 2001
- Strength Pareto Evolutionary Algorithm (SPEA2) - 2002
- Duplicate Elimination Non-domination Sorting Evolutionary Algorithm (DENSEA) - 2006
- Epsilon-Constraint Evolutionary Algorithm (ECEA)
- Fair Evolutionary Multiobjective Optimizer (FEMO)
- Simple Evolutionary Multiobjective Optimizer (SEMO2)
- Multiple Single Objective Pareto Sampling (MSOPS)
- Borg Multi-Objective Evolutionary Algorithm (Borg MOEA) - 2013
- Reference Vector Guided Evolutionary Algorithm (RVEA)
- Estimation of Distribution Algorithm (EDA)
- Strongly Typed Genetic Programming (STGP)
- Constrained Two-Archive Evolutionary Algorithm (C-TAEA) - M - 2019 


### 1.2. Other bio-inspired population-based algorithms


#### 1.2.1. Particle Swarm

- Particle Swarm Optimization (PSO) - S / U
- Particle Swarm Optimization Generational (GPSO) - S / U / Sto
- Non-dominated Sorting Particle Swarm Optimization (NSPSO) - M / U
- Multi-Objective Particle Swarm Optimization (MOPSO or OMOPSO) - 2005
- Speed-constrained Multi-objective Particle Swarm Optimization (SMPSO) - 2009
- Multiswarm Particle Swarm Optimization (MPSO) 


#### 1.2.2. Ant Colony

- Ant Colony Optimisation (ACO)
- Extended Ant Colony Optimization (GACO) - S / C / U / I
- Multi-objective Hypervolume-based Ant Colony Optimisation (MHACO) - M / U / I 

#### 1.2.3. Other

- Grey Wolf Optimizer (GWO) - S / U
- Hunting Search Artificial Bee Colony (ABC) - S / U
- Bees Algorithm Cuckoo Search Memetic algorithm Improved Harmony Search (IHS) - S / M / C / U / I
- Social Cognitive Optimization (SCO) 


### 1.3. Bayesian Optimisation

- Gaussian Process (GP)
- Tree-Structured Parzen Estimator (TPE) 


### 1.4. Brute Force

- Grid Search (GS)


## 2. Local optimisation



![Optimisation_algo_local]({{ site.url }}{{ site.baseurl }}/assets/images/Optimisation_algo_local.png)


### 2.1. Gradient-based

- Hill Climbing (HC)
- Stochastic Hill Climbing (SHC)
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Conjugate Gradient Method (CGD) 
- Method of Moving Asymptotes (MMA) - S / C / U 


### 2.2. Gradient-Free

- Nelder-Mead Simplex (NMS) - S / U
- Tabu Search Simulated Annealing (SA) - S / U 
- Greedy Randomized Adaptive Search Procedure (GRASP) 
- Guided Local Search (GLS) 
- Iterated Local Search (ILS)
- Variable Neighborhood Search (VNS)
- Random Search (RS) 
- Pattern Search (PS)
- DIvided RECTangles (DIRECT) 
