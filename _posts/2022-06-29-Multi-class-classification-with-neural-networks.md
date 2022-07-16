---
title:  "Part 5 : Multi-class classification with neural networks"
excerpt: "Learn to solve a multi-class classification problem with neural networks in Python."
toc: true
category:
  - deep learning
---

https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/


In a previous article, we solved a binary classification on a problem with non-linear decision boundary using a Multi-Layer Perceptron. However, in many mase studies we need to classify observations in more than 2 classes. For example, the famous MNIST dataset is composed of images of handwritten digits and the problem consists in predicting the corresponding digit to each image. In this case, we need to construct a classifier that is capable of assigning values between 0 and 9 to each images, i.e. classify the observations into 10 classes. In this article, we will see how to deal with multi-class classification problems.


## Problem description

Let's consider a problem where we need to classify 2 dimensional data points into 3 classes: 0 (red), 1 (green) and 2 (blue).



## Network architecture

Since our training data in 2 dimensional, we need 2 input neurons. We want to classify our data into 3 classes so we need 3 output neurons. We choose to have a hidden layer composed of 4 neurons. I did not draw the biaises for clarity reason but we will consider them in the derivation.


![marchitecture_multiclass_NN]({{ site.url }}{{ site.baseurl }}/assets/images/architecture_multiclass_NN.png)
<sub><sup>*Architecture of the neural network for multi-class classification*</sup></sub>


## Theory

Based on the network architecture, we can derive the following:

$$
\begin{align*}
z^{1}_0 &= a^{0}_0 w^{1}_0 + a^{0}_1 w^{1}_1 + b^{1}_0\\
&\vdots\\ 
z^{2}_2 &= a^{1}_0 w^{2}_9 + a^{1}_1 w^{2}_{10} + a^{1}_2 w^{2}_{11} + a^{1}_3 w^{2}_{12} + b^{2}_2\\
\end{align*}
$$


AS you can see, it's becoming pretty crammed and cumbersome to write. We will vectorise the equations as follows,

$$
\begin{align*}
Z^{1} &= W^{1}.A^{0} + B^{1} \\
Z^{2} &= W^{2}.A^{1} + B^{2}
\end{align*}
$$

Such as 

$$
\begin{bmatrix} 
z^{1}_0 \\ 
z^{1}_1 \\ 
z^{1}_2 \\ 
z^{1}_3 
\end{bmatrix}

=

\begin{bmatrix} 
w^{1}_0 & w^{1}_1 \\
w^{1}_2 & w^{1}_3 \\
w^{1}_4 & w^{1}_5 \\
w^{1}_6 & w^{1}_7 
\end{bmatrix}
.
\begin{bmatrix} 
a^{0}_0 \\ 
a^{0}_1
\end{bmatrix}
+
\begin{bmatrix} 
b^{1}_0 \\ 
b^{1}_1 \\ 
b^{1}_2 \\ 
b^{1}_3 
\end{bmatrix}
$$

and

$$
\begin{bmatrix} 
z^{2}_0 \\ 
z^{2}_1 \\ 
z^{2}_2
\end{bmatrix}

=

\begin{bmatrix} 
w^{2}_0 & w^{2}_1 & w^{2}_2 & w^{2}_3 \\
w^{2}_4 & w^{2}_5 & w^{2}_6 & w^{2}_7 \\
w^{2}_8 & w^{2}_9 & w^{2}_{10} & w^{2}_{11}
\end{bmatrix}
.
\begin{bmatrix} 
a^{1}_0 \\ 
a^{1}_1 \\ 
a^{1}_2 \\ 
a^{1}_3
\end{bmatrix}
+
\begin{bmatrix} 
b^{2}_0 \\ 
b^{2}_1 \\ 
b^{2}_2
\end{bmatrix}
$$
