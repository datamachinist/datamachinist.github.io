---
title:  "Conda cheat sheet"
excerpt: "Some useful commands and shortcuts for Conda"
toc: true
category:
  - cheat sheet
---


![conda_logo]({{ site.url }}{{ site.baseurl }}/assets/images/datamachinist/conda_logo.png)




## Some definitions

**Conda** is an open source package manager tool and environment management system. It is mostly used for created isolated Python environments but it can package and distribute software for any language.

**Anaconda** is a free and open-source Python distribution for scientific computing. It includes conda and plus hundreds of popular Python packages such as numpy, scipy, matplotlib, pandas, etc...

**Miniconda** is a lighter alternative to Anaconda that just include conda and its dependencies but no Python packages.

A **virtual environment** is a named, isolated, working copy of Python that that maintains its own files, directories, and paths so that you can work with specific versions of libraries or Python itself without affecting other Python projects. Virtual environments make it easy to cleanly separate different projects and avoid problems with different dependencies and version requirements across components.


Please see the [official page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information.


You can download the official Conda cheat sheet [here]({{ site.url }}{{ site.baseurl }}/assets/downloads/conda-cheatsheet.pdf).



## Other environment managers



**virtualenv** is a tool to create isolated environments for Python only.


**venv** is a tool to create isolated environments for Python 3 only.




## Cheat sheet


- Create environment

```bash
conda create -n myenv
conda create -n myenv python=3.6      # with a specific python version
conda create -n myenv scipy           # with a specific python package
conda env create -f environment.yml   # from a .yml file
```


- List environments


```bash
conda env list
```


- Activate environment


```bash
conda activate myenv
```


- Deactivate environment


```bash
conda deactivate
```


- View a list of the packages in an environment


```bash
conda list
conda list -n myenv
```


- Export a YML environment file


```bash
conda env export > environment.yml
```


- Remove an environment


```bash
conda env remove --name myenv
```


- Check conda version


```bash
conda info
```


- Update conda


```bash
conda update conda
```

