---
title:  "Part 2 : The multi-layer perceptron"
excerpt: "We explain the concept behind the backpropagation algorithm with the multi-layer perceptron"
category:
  - deep learning
---


In the [previous article]({% post_url 2022-06-29-the-single-layer-perceptron %}), we learnt the theory behind the simplest neural network possible: the Single-Layer Perceptron. In this post, we will increase the complexity of the network by adding a hidden layer of neurons between the output and the input layer and build a **Multi-Layer Perceptron (MLP)**. We will also introduce the Backpropagation algorithm.



## Architecture of a MLP


We add a hidden layer of neurons between the output and the input layer, as follows:

![machine learning types]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/MLP.png)
<sub><sup>*Architecture of a Multi-Layer Perceptron*</sup></sub>

The layer number is shown into brackets in superscript and the neuron number in each layer is shown in subscript.

## Feedforward

We can now write the activation of each neuron as follows,

$$
a^{(2)}_1 = \sigma(z^{(2)}_1) \\
a^{(1)}_1 = \sigma(z^{(1)}_1) 
$$

where

$$
z^{(2)}_1 = a^{(1)}_1 w^{(1)}_{11}+b^{(1)} \\
z^{(1)}_1 = a^{(0)}_1 w^{(0)}_{11}+b^{(0)} 
$$


## Backpropagation

The Mean-Square Error (MSE) cost function remains the same.

$$
J = \frac{1}{m}\sum_{i=1}^{m}(a^{(2)}_1i-y_i)^2
$$

### Last layer

Let's apply Gradient Descent to the weight and bias of the last layer,

$$
w^{(1)}_{11} := w^{(1)}_{11} - \alpha \frac{dJ}{dw^{(1)}_{11}} \\
b^{(1)} := b^{(1)} - \alpha \frac{dJ}{db^{(1)}} 
$$

The chain rule now becomes,

$$
\frac{dJ}{dw^{(1)}_{11}} = \frac{dJ}{da^{(2)}_1} \frac{da^{(2)}_1}{dz^{(2)}_1} \frac{dz^{(2)}_1}{dw^{(1)}_{11}} 
$$

Isolating each terms, we have:

$$
\frac{dJ}{da^{(2)}_1} = \frac{2}{m}(a^{(2)}_1-y)  \\
\frac{da^{(2)}_1}{dz^{(2)}_1} = \sigma'(z^{(2)}_1) \\
\frac{dz^{(2)}_1}{dw^{(1)}_{11}} = a^{(1)}_1 
$$

Similarly for the bias,

$$
\frac{dJ}{d b^{(1)}} = \frac{dJ}{da^{(2)}_1} \frac{da^{(2)}_1}{dz^{(2)}_1} \frac{dz^{(2)}_1}{d b^{(1)}} 
$$

where $$\frac{dz^{(2)}_1}{d b^{(1)}} = 1$$


The novelty here is that we also need to calculate the **sensitivity** of the cost function with respect to the activation of the previous layer $a^{(1)}_1 $. That's why we call this method **Backpropagation**.

$$
\frac{dJ}{da^{(1)}_1} = \frac{dJ}{da^{(2)}_1} \frac{da^{(2)}_1}{dz^{(2)}_1} \frac{dz^{(2)}_1}{da^{(1)}_1} 
$$

The only extra term we need to calculate is:

$$
\frac{dz^{(2)}_1}{da^{(1)}_1} = w^{(1)}
$$

### First layer

We can now look at adjusting the weight and bias of the first layer by using the same idea.

Gradient descent for the weight in the first layer:

$$
w^{(0)}_{11} := w^{(0)}_{11} - \alpha \frac{dJ}{dw^{(0)}_{11}}
$$

Chain rule for the weight in the first layer:

$$
\frac{dJ}{dw^{(0)}_{11}} = \frac{dJ}{da^{(1)}_1} \frac{da^{(1)}_1}{dz^{(1)}_1} \frac{dz^{(1)}_1}{dw^{(0)}_{11}} 
$$

where

$$
\frac{dJ}{da^{(1)}_1} = \textrm{already calculated} \\
\frac{da^{(1)}_1}{dz^{(1)}_1} = \sigma'(z^{(1)}_1) \\
\frac{dz^{(1)}_1}{dw^{(0)}_{11}} = a^{(0)}_1 
$$

Gradient descent for the bias in the first layer:

$$
b^{(0)} := b^{(0)} - \alpha \frac{dJ}{db^{(0)}} 
$$

Chain rule for the bias in the first layer:

$$
\frac{dJ}{db^{(0)}} = \frac{dJ}{da^{(1)}_1} \frac{da^{(1)}_1}{dz^{(1)}_1} \frac{dz^{(1)}_1}{db^{(0)}} 
$$

where $$\frac{dz^{(1)}_1}{db^{(0)}} = 1$$


## Conclusion

We now have all the bits and pieces to update the weights and biases of all the layers. We won't look at the implementation into Python code just yet because this example is a bit meaningless in practice since there is only one input neuron i.e. one feature. (So the best classification we can get in this case is input = output). In the next article, we will look at implementing a single-layer neural network with 3 input neurons. 









