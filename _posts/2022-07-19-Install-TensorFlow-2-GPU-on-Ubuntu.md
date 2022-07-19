---
title:  "Part 8 : Install TensorFlow 2 GPU support on Ubuntu"
excerpt: "A step-by-step instructions to install Tensorflow2 with GPU support on Ubuntu"
toc: true
category:
  - deep learning
---

Installing Tensorflow 2 with GPU support can be a challenge. I will give step-by-step instructions to perform this installation on Ubuntu.



## 1.CUDA pre-installation steps

- Take note of your GPU brand and make

```bash
lspci | grep -i nvidia
# OR
sudo lshw -C display
```

- Verify you have a CUDA-Capable GPU by checking if it is listed [here](https://developer.nvidia.com/cuda-gpus).

- Verify you have a supported version of Linux

```bash
uname -m && cat /etc/*release
```
You should check that you are running on a 64-bit system (`x86_64`).

- Verify the system has `gcc` installed

```bash
gcc --version
```

- Verify the system has correct Linux kernel headers

```bash
# list the Linux kernel
uname -r  
# Install the Linux kernel hearders
sudo apt-get install linux-headers-$(uname -r)
```


## 2.CUDA Toolkit installation

- Download the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
```

- Calculate the MD5 checksum of the downloaded file and compare it to [them](https://developer.download.nvidia.com/compute/cuda/11.7.0/docs/sidebar/md5sum.txt)

```bash
md5sum cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
```

- Install the repository meta-data

```bash
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
```

- Update the GPG key and the apt-get cache

```bash
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
```

- Install CUDA

```bash
sudo apt-get -y install cuda
```

- Reboot the system to load the NVIDIA drivers

```bash
sudo reboot
```

## 3.CUDA post-installation steps

- Export CUDA environment variables to the $PATH (add them to `.bashrc`) 

```bash
echo 'export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

- Check whether the NVIDIA Persistence Daemon is active

```bash
systemctl status nvidia-persistenced
```

- Disable the udev rule because it could interfere with the driver

```bash
# copy the udev rule
sudo cp /lib/udev/rules.d/40-vm-hotadd.rules /etc/udev/rules.d

# edit the udev rule
sudo vim /etc/udev/rules.d/40-vm-hotadd.rules
```

Comment out this line:

```bash
SUBSYSTEM=="memory", ACTION=="add", DEVPATH=="/devices/system/memory/memory[0-9]*", TEST=="state", ATTR{state}!="online", ATTR{state}="online"
```

- Install [CUDA-samples](https://github.com/nvidia/cuda-samples) and run the `deviceQuery` sample

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery/
make
./deviceQuery
```

If the installation went well, you should see some information about CUDA.

- Verify the Driver, NVCC and CUDA installation

```bash
cat /proc/driver/nvidia/version
nvcc -V
nvidia-smi
```

- *Optional*: Install third-party libraries

```bash
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev
```

## 4. Install CuDNN

- Download cuDNN at this [link](https://developer.nvidia.com/rdp/cudnn-download)

You will be asked to create a NVIDIA account. Download the `.deb` file corresponding to your Ubuntu and CUDA version.

- Install the local repositoru

```bash
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.4.1.50_1.0-1_amd64.deb 
```

- Import the CUDA GPG key

```bash
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
```

- Refresh the repository metadata.

```bash
sudo apt-get update
```

- Install the runtime library.

```bash
sudo apt-get install libcudnn8=8.4.1.50-1+cuda11.6
```
- Install the developer library.

```bash
sudo apt-get install libcudnn8-dev=8.4.1.50-1+cuda11.6
```

- Install the code samples and the cuDNN library documentation.

```bash
sudo apt-get install libcudnn8-samples=8.4.1.50-1+cuda11.6
```

## 5. Verify CuDNN installation

- Copy the cuDNN samples to a writable path.

```bash
$cp -r /usr/src/cudnn_samples_v8/ $HOME
```

- Go to the writable path.

```bash
$ cd  $HOME/cudnn_samples_v8/mnistCUDNN
```

- Compile the mnistCUDNN sample.

```bash
$make clean && make
```

- Run the mnistCUDNN sample.

```bash
$ ./mnistCUDNN
```

If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following: `Test passed!`


## 6. Install and test Tensorflow 2 with GPU support

```bash
pip install tensorflow
```

Open a Python shell

```bash
python
```

And type the follow:

```python
import tensorflow as tf
tf.config.list_physical_devices("GPU")
```

If the installation went well, you should see `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

Congratulations, you completed the installation!