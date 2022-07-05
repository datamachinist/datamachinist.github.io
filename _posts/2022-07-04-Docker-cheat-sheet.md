---
title:  "Docker cheat sheet"
excerpt: "Some useful commands and shortcuts for Docker"
toc: true
category:
  - cheat sheet
---


![docker_logo]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/docker_logo.png)

## What is Docker?

[Docker](https://www.docker.com/) is a tool used to deploy software across different platform without running into compatibility issues.

A Docker **image** is a lightweight, standalone, executable package of software that includes everything needed to run an application (code, libraries, settings, etc...).  On a Linux system, Docker images are stored in /var/lib/docker.

A running instance of an image is called a **container**. A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another.

## Cheat sheet

You can download a printable Linux command line cheat sheet [here]({{ site.url }}{{ site.baseurl }}/assets/downloads/docker-cheat-sheet.pdf) from [Cheatography](https://cheatography.com/gambit/cheat-sheets/docker/).

Another handy cheat sheet can be found [here](https://github.com/wsargent/docker-cheat-sheet).


- Check Docker installation


```bash
docker version
docker run hello-world
```


- List docker images


```bash
docker images
```


- List dangling Docker images


```bash
docker images -f dangling=true
```


- Remove dangling images and build caches, all stopped containers and all networks not used by at least 1 container (useful to free some space).


```bash
docker system prune
```


- Remove docker image by ID


```bash
docker rmi -f IMAGE_ID
```

- Download a Docker image


```bash
docker pull IMAGE_NAME
```


- List running containers


```bash
docker ps
docker container ls
```


- List all containers


```bash
docker ps -a
```


- Stop a container


```bash
docker stop CONTAINER_ID
```


- Remove a container


```bash
docker rm CONTAINER_ID
```


- Run a docker image


```bash
docker run -it --rm IMAGE_ID
```


- Options of the run command:


```bash
-u $(id -u):$(id -g)       # assign a user and a group ID
--gpus all                 # allow GPU support
-it                        # run an interactive container inside a terminal
--rm                        # automatically remove the container after exiting
--name my_container        # give it a friendly name
-v ~/docker_ws:/notebooks  # share a directory between the host and the container
-p 8888:8888               # define port 8888 to connect to the container (for Jupyter notebooks)
```


- Open a new terminal in a running container


```bash
docker exec -it CONTAINER_ID bash
```


## Create a new Docker image



There are 2 main methods.



**Method 1 : Using Dockerfile**



1. Create a file called Dockerfile from your host machine.


```bash
vim Dockerfile
```

```bash
# getting base Ubuntu image
FROM ubuntu

# file author / maintainer
LABEL maintainer=<your_email_address>;

# update the repository sources list
RUN apt-get update

# print Hello World
CMD ["echo", "Hello World"]
```


2. Build the image.


```bash
docker build -t myimage:0.1 .
```


3. Run the image.


```bash
docker run myimage:0.1
```


**Method 2 : Commit a Docker image from a running container**


Modify a running container and run this in another terminal.


```bash
docker commit CONTAINER_ID my_new_image
```

