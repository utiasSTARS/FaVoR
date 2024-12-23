Installation
============

Follow the steps below to install and set up the environment for the project.

Requirements
------------

To run the project, you need at least Python 3.8.15 and the dependencies specified in the `requirements.txt` file.

Installation
------------
Clone the repository and install the requirements.

```bash
$ git clone --recursive https://github.com/utiasSTARS/FaVoR.git
$ cd FaVoR
```

Use Conda to create a new environment and install the requirements.

```bash
$ conda create -n favor python=3.10
$ conda activate favor
$ conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```

```bash
$ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
$ pip install -r requirements.txt
```

Now build the cuda modules as:

```bash
$ cd lib/cuda
$ ./build.sh
```