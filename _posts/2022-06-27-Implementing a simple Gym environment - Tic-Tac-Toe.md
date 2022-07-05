---
title:  "Part 8.2 : Implementing a simple Gym environment - Tic-Tac-Toe"
excerpt: "Implement the 4 methods for the tic-tac-toe Gym environment."
toc: true
category:
  - reinforcement learning
  - custom Gym environment
---


In the previous article, we have created, installed and registered a minimalist Gym environment. However, this environment was not doing anything since we didn't implement the  4 methods of the environment class: __init__, step, reset and render. In this article, we will see how to implement these 4 methods for a simple game: the tic-tac-toe.

Let's remind ourselves the rules of the game. The game is played on a grid that's 3 squares by 3 squares. There are 2 players, one with X and the other with O. Players take turns putting their marks in empty squares. The first player to get 3 of her marks in a row (up, down, across, or diagonally) is the winner. A reward of +100 is given to the player wining the game.


![tic-tac-toe]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/Tic_tac_toe.png)
<sub><sup>*The tic-tac-toe game*</sup></sub>

Here is an example of implementation of the 4 methods.

```python
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TicTacEnv(gym.Env):
	metadata = {'render.modes': ['human']}


	def __init__(self):
		self.state = []
		for i in range(3):
			self.state += [[]]
			for j in range(3):
				self.state[i] += ["-"]
		self.counter = 0
		self.done = 0
		self.add = [0, 0]
		self.reward = 0

	def check(self):

		if(self.counter<5):
			return 0
		for i in range(3):
			if(self.state[i][0] != "-" and self.state[i][1] == self.state[i][0] and self.state[i][1] == self.state[i][2]):
				if(self.state[i][0] == "o"):
					return 1
				else:
					return 2
			if(self.state[0][i] != "-" and self.state[1][i] == self.state[0][i] and self.state[1][i] == self.state[2][i]):
				if(self.state[0][i] == "o"):
					return 1
				else:
					return 2
		if(self.state[0][0] != "-" and self.state[1][1] == self.state[0][0] and self.state[1][1] == self.state[2][2]):
			if(self.state[0][0] == "o"):
				return 1
			else:
				return 2
		if(self.state[0][2] != "-" and self.state[0][2] == self.state[1][1] and self.state[1][1] == self.state[2][0]):
			if(self.state[1][1] == "o"):
				return 1
			else:
				return 2

	def step(self, target):
		if self.done == 1:
			print("Game Over")
			return [self.state, self.reward, self.done, self.add]
		elif self.state[int(target/3)][target%3] != "-":
			print("Invalid Step")
			return [self.state, self.reward, self.done, self.add]
		else:
			if(self.counter%2 == 0):
				self.state[int(target/3)][target%3] = "o"
			else:
				self.state[int(target/3)][target%3] = "x"
			self.counter += 1
			if(self.counter == 9):
				self.done = 1;

		win = self.check()
		if(win):
			self.done = 1;
			print("Player ", win, " wins.", sep = "", end = "\n")
			self.add[win-1] = 1;
			if win == 1:
				self.reward = 100
			else:
				self.reward = -100

		return [self.state, self.reward, self.done, self.add]

	def reset(self):
		for i in range(3):
			for j in range(3):
				self.state[i][j] = "-"
		self.counter = 0
		self.done = 0
		self.add = [0, 0]
		self.reward = 0
		return self.state

	def render(self):
		for i in range(3):
			for j in range(3):
				print(self.state[i][j], end = " ")
			print("")
```

This code is largely based on [this article](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa). The code can be found on [GitHub](https://github.com/PierreExeter/gym-tictac).

As previously, we install and register the environment.

```bash
pip install -e .
```

We can test the environment using this code.

```python
import gym
import gym_tictac

env = gym.make('tictac-v0')

for e in range(3):
     env.reset()
     print("######")
     print("EPISODE: ", e)
     print("######")

     for t in range(9):
          env.render()
          action = t
          state, reward, done, info = env.step(action) 
          print("reward: ", reward)
          print("")

 env.close()
```

You should see the following output.

```bash
######
EPISODE:  0
######
 - - -
 - - -
 - - -
 reward:  0
 ****** 
 o - - 
 - - -
 - - -
 reward:  0
 ******
 o x - 
 - - -
 - - -
 reward:  0
 ******
 o x o 
 - - -
 - - -
 reward:  0
 ******
 o x o 
 x - - 
 - - -
 reward:  0
 ******
 o x o 
 x o - 
 - - -
 reward:  0
 ******
 o x o 
 x o x 
 - - -
 Player 1 wins.
 reward:  100
 ******
 o x o 
 x o x 
 o - - 
 Game Over
 reward:  100
 ******
 o x o 
 x o x 
 o - - 
 Game Over
 reward:  100
 ******
```

This is not very exciting, as each player is adding its token one after the other but this is just to illustrate how to use the environment. In the next article, we will see how to create a more interesting Gym environment using the Pybullet physics engine.