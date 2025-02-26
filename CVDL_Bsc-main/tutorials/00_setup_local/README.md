# Setup Instructions for Local Development

## Introduction

In this tutorial, you will setup the environment for the practical course
elements. This includes:
* Setup the course Git repository on ZHAW Github
* Setup your integrated development environment (IDE)
* Run example code

Note, the choice of IDE and remote development tools is up to you.
We provide guidance and limited support for Visual Studio Code (VSCode).

## Step I: Setup Git

### SSH authentication

We recommend using SSH to access the Git repository, because it is secure and
convinient. Navigate to https://github.zhaw.ch/settings/keys and add your SSH
key. If this is your first time using SSH keys, follow the
[GitHub guide on how to generate keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
first. For this to work on ZHAW GitHub, you will need will need to change the
`ssh config` entry to the following:

```text
Host github.zhaw.ch
    HostName github.zhaw.ch
    User git
    IdentityFile ~/.ssh/YOUR_PRIVATE_KEY
```

To test that your ssh keys work as authentication, run the following command in
your terminal:

```bash
ssh -T git@github.zhaw.chh
```

### clone repo

When you have added your SSH key to the ZHAW Github and configured `~/.ssh/config`, you can clone the course repository.

Clone the repository:

```bash
git clone git@github.zhaw.ch:CVDL/CVDL_BSc.git
```

> **_NOTE_**
> We recommend you fork the repository as it allows you to commit to it and collaborate in a group

## Step II: Setup Python

All the labs have been tested with **Python-3.12** so preferably you would use it as well. you can install a specific version from [Python's website](https://www.python.org/downloads/) or use a tool like [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

## Step III: Setup Python environment

We recommend you use the `venv` Python module as its shipped with the Python standard library, so you don't need to install anything to create a virtual environment. If you are using conda, you should use it's native environment capabilities.

### Create a venv environment for this course

Creating, activating, and deactivating a virtual environment is done with the following commands:
```bash
# Create an environment in the folder .cvdl with the default python interpreter
python3 -m venv .cvdl
# or specify the python interpreter
/path/to/interpreter/python -m venv .cvdl

# Activate the environment
source .cvdl/bin/activate

# Deactivate the environment
deactivate
```

## Step IV: Install Course dependencies

Activate/Source your python env and install the dependencies for this course.
```bash
# Change into the repository if you are not already there
cd CVDL_BSc

# Activate venv
source .cvdl/bin/activate

# Install the course dependencies
pip install -r requirements.txt
```

### Verify the setup
In the cvdl env launch python and check imports.

```bash
# venv
source .cvdl/bin/activate

# run python snippet
python -c "import torch; print(torch.__version__)"
```

## Step V: Setup IDE

We recommend using VSCode as its open-core, fast and has support for python.

Download and install VSCode from [the website](https://code.visualstudio.com/).
Open VSCode and install the recommended extensions:

* [python extension package](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* [pylance language server](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
* [jupyter extension pack](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
* [remote - ssh](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)

open the [quickstart script](quickstart.py) adaptet from [pytorch](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html), select cvdl as the python interpreter and run the whole script.

## Step VI: Setup Remote Development

* See the [tutorial 01](../01_setup_lightning/README.md) on how to setup remote development on lightning.ai.
* See the [tutorial 02](../02_setup_colab/README.md) on how to setup remote development on Google Colab.
