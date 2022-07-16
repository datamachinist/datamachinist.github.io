---
title:  "Part 7 : Install TensorFlow 2 with GPU support using Docker on Ubuntu"
excerpt: "Installing TensorFlow with GPU support on Ubuntu can be troublesome. We will see how to use Docker avoid a headache."
toc: true
category:
  - deep learning
---


![tensorflow_docker]({{ site.url }}{{ site.baseurl }}/assets/images/logo_all.png)

In previous posts, we have built simple neural networks by hand. Fortunately, there are libraries to build network architectures and calculate gradients automatically. TensorFlow is one of the most famous one. I will explain how to install this Python library on Ubuntu 18.04.

## Why using Docker?

Neural network calculations are primarily based on matrix operations, which are most efficiently performed on GPUs. In order to use your computer's GPU with TensorFlow, it is necessary to install 2 libraries on your machine:
- **CUDA** (Compute Unified Device Architecture): a parallel computing platform developed by NVIDIA for general computing on GPUs
- **cuDNN** (CUDA Deep Neural Network): a GPU-accelerated library of primitives used to accelerate deep learning frameworks such as TensorFlow or Pytorch.

In order to be able to use these libraries, you must first ensure that your computer is equipped with a CUDA-enabled GPU. The list of these GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). You must then install the latest NVIDIA driver corresponding to your GPU.


As you can see, there is a lot of prerequisites before being able to install TensorFlow. You can follow the official procedure to install CUDA from the NVIDIA website [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). However, I learnt the hard way that it is easy to mess up your computer and your graphics card while installing all these libraries and drivers. That's why, I would highly recommend installing TensorFlow inside a [Docker](https://www.docker.com/) container.

Docker is essentially a self-contained OS with all the dependencies necessary for a smooth installation.

## Let's install!

First of all, check the instructions on the official TensorFlow [page](https://www.tensorflow.org/install/docker).

### 1.Install the latest NVIDIA drivers

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 2. Install Docker

Please follow [these instructions](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

Check that you have installed Docker 19.03 or higher.

```bash
docker version
```

Add the current user to the docker group and reboot.

```bash
sudo usermod -a -G docker $USER
sudo reboot
```

Test docker

```bash
docker run hello-world
```

### 3. Install the NVIDIA Container toolkit

Please follow [these instructions](https://github.com/NVIDIA/nvidia-docker).

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Test that everything was installed correctly.

```bash
docker run --gpus all --rm nvidia/cuda nvidia-smi
```

You should see some information about your GPU and the CUDA version installed.


### 4. Download the TensorFlow Docker images with GPU support


```bash
docker pull tensorflow/tensorflow:latest-gpu-py3
docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter
```

### 5. Test that the image is working properly


```bash
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu-py3 python -c "import tensorflow as tf; print(tf.version); print(tf.test.is_gpu_available()); print(tf.test.is_built_with_cuda())"
```

This should return the TensorFlow version and whether GPU support is available.


Please have a look at my [Docker cheat sheet]({% post_url 2022-07-04-Docker-cheat-sheet %}) for more information about Docker.




### 6. Run a TensorFlow container



Create a new container from the TensorFlow image


```bash
docker run -it --rm tensorflow/tensorflow:latest-gpu-py3
```


You should be logged-in in the new container. You can explore it using ls, cd, etc... You can exit using $ exit. Now let's see a more practical example. First, let's create a directory to exchange files between your machine and the container:


```bash
mkdir ~/docker_ws
```

```bash
docker run -u $(id -u):$(id -g) --gpus all -it --rm --name my_tf_container -v ~/docker_ws:/notebooks -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu-py3-jupyter
```


 Let's explain the different options.


```bash
-u $(id -u):$(id -g)       # assign a user and a group ID
--gpus all                 # allow GPU support
-it                        # run an interactive container inside a terminal
-rm                        # automatically clean up the container and remove the file system after closing the container
--name my_tf_container     # give it a friendly name
-v ~/docker_ws:/notebooks  # share a directory between the host and the container
-p 8888:8888               # define port 8888 to connect to the container
-p 6006:6006               # forward port 6006 for Tensorboard
```


Once the container is running, your should see a URL to copy and paste in your browser that looks like "http://127.0.0.1:8888/?token=xxxxxxxxxx". You should then see a list of TensorFlow tutorials, as shown below.

![tf_tutorials]({{ site.url }}{{ site.baseurl }}/assets/images/tf_tutorials.png)
<sub><sup>*Tensorflow tutorials*</sup></sub>

Finally, you can use $ docker exec to run a command inside a running docker container. In another terminal, run this command:


```bash
docker exec -it my_tf_container tensorboard --logdir tf_logs/
```


You should be able to access the TensorBoard page via this URL "http://localhost:6006/" (see also [this tutorial](https://www.youtube.com/watch?v=W3bk2pojLoU))


Play around with the tutorials and enjoy!


