---
title:  "Part 3 - Optimisation libraries"
excerpt: "A comparison of some popular optimisation frameworks"
toc: true
category:
  - optimisation
---


In this article, we will compare the following optimisation frameworks:
- [Platypus](https://github.com/Project-Platypus/Platypus)
- [MOEA](https://github.com/MOEAFramework/MOEAFramework)
- [PyGMO](https://github.com/esa/pygmo2)
- [PaGMO](https://github.com/esa/pagmo2)
- [Inspyred](https://github.com/aarongarrett/inspyred)
- [DEAP](https://github.com/DEAP/deap)
- [EMOO](https://projects.g-node.org/emoo/)
- [jMetal](https://github.com/jMetal/jMetalPy)
- [PYMOO](https://github.com/anyoptimization/pymoo)



## Features comparison

| Features            | Platypus | MOEA | PyGMO  | PaGMO | Inspyred | DEAP   | EMOO   | jMetalPy | PYMOO  |
|---------------------|----------|------|--------|-------|----------|--------|--------|----------|--------|
| Language            | Python   | Java | Python | C++   | Python   | Python | Python | Python   | Python |
| Open source         | X        | X    | X      | X     | X        | X      | X      | X        | X      |
| Parallelisation     | X        | X    | X      | X     |          |        |        |          |        |
| Documentation       | X        | X    | X      | X     | X        | X      |        | X        | X      |
| Constrained PB      | X        | X    | X      |       |          |        |        |          |        |
| Unconstrained PB    | X        | X    | X      |       |          |        |        |          |        |
| Multi-objective PB  | X        | X    | X      |       |          |        |        |          |        |
| Single-objective PB |          |      | X      |       |          |        |        |          |        |
| Continuous PB       |          |      | X      |       |          |        |        |          |        |
| Integer PB          |          |      | X      |       |          |        |        |          |        |
| Stochastic PB       |          |      | X      |       |          |        |        |          |        |
| Deterministic PB    |          |      | X      |       |          |        |        |          |        |

## Algorithms comparison


| Algorithm      | Platypus / MOEA | PyGMO / PaGMO | Inspyred | DEAP | EMOO | jMetalPy / jMetal | PYMOO |
|----------------|-----------------|---------------|----------|------|------|-------------------|-------|
| NSGA-II        | X               | X             | X        | X    |      | X                 | X     |
| NSGA-III       | X               |               |          | X    |      | X                 | X     |
| R-NSGA-III     |                 |               |          |      |      |                   | X     |
| U-NSGA-III     |                 |               |          |      |      |                   | X     |
| G-NSGA-II      |                 |               |          |      |      | X                 |       |
| R-NSGA-II      |                 |               |          |      |      |                   | X     |
| MOEA/D         | X               | X             |          |      |      | X                 | X     |
| MOEA/D-DRA     |                 |               |          |      |      | X                 |       |
| IBEA           | X               |               |          |      |      | X                 |       |
| ε-MOEA         | X               |               |          |      |      | X                 |       |
| SPEA2          | X               |               |          | X    |      | X                 |       |
| G-SPEA2        |                 |               |          |      |      | X                 |       |
| GDE3           | X               |               |          |      |      | X                 |       |
| G-GDE3         |                 |               |          |      |      | X                 |       |
| OMOPSO         | X               |               |          |      |      | X                 |       |
| SMPSO          | X               |               |          |      |      | X                 |       |
| G-SMPSO        |                 |               |          |      |      | X                 |       |
| ε-NSGA II      | X               |               |          |      |      |                   |       |
| CMA-ES         | X               | X             |          | X    |      |                   | X     |
| PESA2          | X               |               |          |      |      |                   |       |
| SMS-EMOA       | X               |               |          |      |      |                   |       |
| PAES           | X               |               | X        |      |      |                   |       |
| AbySS          | X               |               |          |      |      |                   |       |
| Borg MOEA      | X               |               |          |      |      |                   |       |
| CellDE         | X               |               |          |      |      |                   |       |
| DBEA           | X               |               |          |      |      |                   |       |
| DE             | X               | X             |          |      |      |                   | X     |
| DENSEA         | X               |               |          |      |      |                   |       |
| ECEA           | X               |               |          |      |      |                   |       |
| ES             | X               |               | X        |      |      |                   |       |
| FastPGA        | X               |               |          |      |      |                   |       |
| FEMO           | X               |               |          |      |      |                   |       |
| GA             | X               | X             | X        |      |      |                   | X     |
| HypE           | X               |               |          |      |      | X                 |       |
| MoCell         | X               |               |          |      |      |                   |       |
| MOCHC          | X               |               |          |      |      |                   |       |
| MSOPS          | X               |               |          |      |      |                   |       |
| Random         | X               |               |          |      |      |                   |       |
| RSO            | X               |               |          |      |      |                   |       |
| RVEA           | X               |               |          |      |      |                   |       |
| SEMO2          | X               |               |          |      |      |                   |       |
| SHV            | X               |               |          |      |      |                   |       |
| SIBEA          | X               |               |          |      |      |                   |       |
| SMPSO          | X               |               |          |      |      |                   |       |
| VEGA           | X               |               |          |      |      |                   |       |
| GACO           |                 | X             |          |      |      |                   |       |
| jDE            |                 | X             |          |      |      |                   |       |
| iDE            |                 | X             |          |      |      |                   |       |
| pDE            |                 | X             |          |      |      |                   |       |
| DE             |                 |               |          | X    |      |                   |       |
| GWO            |                 | X             |          |      |      |                   |       |
| IHS            |                 | X             |          |      |      |                   |       |
| PSO            |                 | X             | X        | X    |      |                   | X     |
| GPSO           |                 | X             |          |      |      |                   |       |
| (N+1)ES        |                 | X             |          |      |      |                   |       |
| ABC            |                 | X             |          |      |      |                   |       |
| SA             |                 | X             | X        |      |      |                   |       |
| xNES           |                 | X             |          |      |      |                   |       |
| MHACO          |                 | X             |          |      |      |                   |       |
| NSPSO          |                 | X             |          |      |      |                   |       |
| DEA            |                 |               | X        |      |      |                   |       |
| EDA            |                 |               | X        | X    |      |                   |       |
| ACO            |                 |               | X        |      |      |                   |       |
| GP             |                 |               |          | X    |      |                   |       |
| MO-CMA-ES      |                 |               |          | X    |      |                   |       |
| STGP           |                 |               |          | X    |      |                   |       |
| BI-POP CMA-ES  |                 |               |          | X    |      |                   |       |
| Multiswarm PSO |                 |               |          | X    |      |                   |       |
| Nelder-Mead    |                 |               |          |      |      |                   | X     |
| Pattern search |                 |               |          |      |      |                   | X     |
| BRK-GA         |                 |               |          |      |      |                   | X     |
| C-TAEA         |                 |               |          |      |      |                   | X     |

