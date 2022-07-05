---
title:  "Part 8.1 : Registering a custom Gym environment"
excerpt: "Learn to create and register a minimal custom Gym environment."
toc: true
category:
  - reinforcement learning
  - custom Gym environment
---


In this tutorial, we will create and register a minimal gym environment. Please [read the introduction](/_posts/2022-06-24-create-training-environments-with-openAI-gym.md) before starting this tutorial.

First of all, let's understand what is a Gym environment exactly. A Gym environment contains all the necessary functionalities to that an agent can interact with it. Basically, it is a class with 4 methods:
- **__init__**: initialization function of the class
- **step**: it takes an action argument and returns a list of 4 items: the next state, the reward of the current state, a boolean representing whether the current episode of our model is done and some additional info on our problem
- **reset**: resets the state and other variables of the environment to the start state
- **render**: displays the environment graphically.

All Gym environments come in a PIP package with the following file structure.

```bash
gym-foo/
  README.md
  setup.py
  gym_foo/
    __init__.py
    envs/
      __init__.py
      foo_env.py
```

Let's explain each items one by one.

- **gym-foo/**
This is the main directory where the environments live.

- **gym-foo/README.md**
This is a short description of the environment.

- **gym-foo/setup.py**

This is the PIP installation file. It describes the name, version and required dependencies of our environment.

```python
from setuptools import setup

setup(name='gym_foo',
       version='0.0.1',
       install_requires=['gym']
 )
```

- **gym-foo/gym_foo/__init__.py**

This file defines the the environment version name and entry points. Make sure you respect the spelling.

```python
from gym.envs.registration import register

register(
     id='foo-v0',
     entry_point='gym_foo.envs:FooEnv',
 )
```

- **gym-foo/gym_foo/envs/__init__.py**

This file contains the following:

```python
from gym_foo.envs.foo_env import FooEnv
```

- **gym-foo/gym_foo/envs/foo_env.py**

This is where the environment class and its 4 methods are described. Since this is a minimal working example, these methods will be left empty for now.

```python
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass
  
    def step(self, action):
        pass
  
    def reset(self):
        pass
  
    def render(self, mode='human', close=False):
        pass
```

We are now ready to install and register the environment. Go back to the initial directory gym-foo and run this command to install the PIP package.

```bash
pip install -e .
```

The -e option means that the package is installed in "editable" mode. This means that the package is installed locally and that any changes to the original package will reflect directly in your environment.

Finally, we can test that everything is working fine by creating one last file: **test_gym_foo.py**

```python
import gym
import gym_foo

env = gym.make('foo-v0')
```

If you are able to run this without error, congratulations you have successfully registered your first Gym environment! The code for this tutorial can be found on [GitHub](https://github.com/PierreExeter/gym-foo).

Note that you can list all the environments registered in Gym using the following Python code.
```python
from gym import envs
import gym_foo

envids = [spec.id for spec in envs.registry.all()]
for envid in sorted(envids):
    print(envid)
```

In the next article, we will implement the 4 methods to create a simple Tic Tac Toe environment.