# mlops-basics
This repository contains a basc working example of performing MLOps. This is implemented by following the MLOps tutorials from [here](https://github.com/graviraja/MLOps-Basics).

## Contents
* [Directory Structure](directory-structure)
* [Installation Instructions](installation-instructions)
  * [Local Setup](local-setup)
  * [Docker Setup](docker-setup)
* [Usage Instructions](usage-instructions)

## Directory Structure

```
mlops-basics
│   README.md
└───docker
└───src
└───tests
└───scripts
```

The `docker` directory contains the docker file to build a docker container. `src` folder contains the model training code. `tests` directory contains unittests to test the model inferencing and training. `scripts` directory contains the scripts to setup the model inferencing.

## Installation Instructions
Following are the installation instructions for setting up the mlops-basics repository:

### Local Setup
These are the steps that need to be performed to setup a local Python environment in a Linux system:
* Install `miniconda` by downloading the installer for Python 3.8 from this [link](https://docs.conda.io/en/latest/miniconda.html) and running the following command:
  ```sh
  $ wget {miniconda installer link}
  $ chmod +x Miniconda3-latest-Linux-x86_64.sh
  $ ./Miniconda3-latest-Linux-x86_64.sh
  ```
  Review the license, accept the terms and setup default installation path to complete the installation.
* Open a new terminal with miniconda setup. You should see the prefix `(base)` in the command line for successful installation. Create a new virtual environment `mlops` for the project and activate it:
  ```sh
  (base) $ conda create -n mlops python=3.8
  (base) $ conda activate mlops
  (mlops) $
  ```
* Install `pytorch` and its dependencies using conda installer:
  ```sh
  (mlops) $ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
  ```
* Install the python dependencies by installing the requirements in the `requirements.txt` file:
  ```sh
  $ pip install -r requirements.txt
  ```

## Usage Instructions
Following are the usage instructions for using the mlops-basics repository