---
title:  "Part 1 : The single-layer perceptron"
excerpt: "In this post, we will explain how simple artificial neural networks works"
category:
  - deep learning
---

The Artificial Neural Network (ANN) is the main algorithm used in deep learning. It is a supervised machine learning technique that can be used for classification or regression problems. We will focus here on classification problems only. These algorithms mimics the structure and function of biological neural networks. In this post, we will describe the theory behind the ANNs. 

[Here](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.28570&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) is a nice visualisation of what's happening in an ANN during training. It lets you choose the input dataset, the type of problem (classification or regression), the training parameters (learning rate, activation function, regularization, etc...) and visualise the output.



## Structure


The simplest form of ANN is called **Feedforward Neural Network (FNN)**, where a number of simple processing units (neurons) are organized in layers. Every neuron in a layer is connected with all the neurons in the previous layer. These connections are not all equal: each connection may have a different strength or weight. The weights on these connections encode the knowledge of a network. An FNN with one layer is called a Single-Layer Perceptron (SLP), an FNN with more than one layer is called... a Multi-Layer Perceptron.

We will start very simple and explain how the Single-Layer Perceptron works for a classification problem. The structure of the SLP is as follows,



![machine learning types](/assets/images/datamachinist/SLP.png)
<sub><sup>*Structure of a Single-Layer Perceptron*</sup></sub>


The SLP maps an **input** <img src="https://render.githubusercontent.com/render/math?math=x"> (there is only one feature here) to a predicted **output** <img src="https://render.githubusercontent.com/render/math?math=\hat{y}">. We define a **weight** <img src="https://render.githubusercontent.com/render/math?math=w"> and a **bias** <img src="https://render.githubusercontent.com/render/math?math=b"> between the 2 layers. The goal is to determine the weight and the bias that minimise a cost function that we will define later.

$\hat{y}$ is the output of an activation function such as the Sigmoid function. This function squashes a real number into the [0, 1] interval. If the real number is negative, the output is close to 0 and if it is positive, the output is close to 1.



<img src="https://latex.codecogs.com/gif.latex?P(s | O_t )=\text { Probability of a sensor reading value when sleep onset is observed at a time bin } t " />


When $a \ne 0$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are 
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \l
