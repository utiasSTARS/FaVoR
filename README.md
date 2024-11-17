<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h2 align="center">The Best of Both the Worlds:<br>Sparse Voxelixed Feature Renderer
</h2>

  <p align="center">
A feature renderer for 3D feature points representation for robust robotics localization.
    <br/>
    <a href="https://papers.starslab.ca/favor/">Webpage</a>
    ·
    <a href="https://github.com/utiasSTARS/FaVoR/issues">Report Bug</a>
    ·
    <a href="https://github.com/utiasSTARS/FaVoR/issues">Request Feature</a>
  </p>
</div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache 2.0 License][license-shield]][license-url]

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li><a href="#structure">Structure</a></li>
    <li><a href="#usage">Usage</a></li>
        <ul><a href="#requirements">Requirements</a></ul>
        <ul><a href="#installation">How to Run the Code</a></ul>
        <ul><a href="#docker">Docker</a></ul>
        <ul><a href="#interface">Interface</a></ul>
        <ul><a href="#demo">Demo</a></ul>
    <li><a href="datasets">Datasets</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About

<div align="center">
  <a href="https://github.com/utiasSTARS/FaVoR">
    <img src="images/FaVoR.gif" alt="demo" >
  </a>
</div>

If you use any of this code, please cite the following publication:

```bibtex
@misc{polizzi2024arXiv,
    title={FaVoR: Features via Voxel Rendering for Camera Relocalization}, 
    author={Vincenzo Polizzi and Marco Cannici and Davide Scaramuzza and Jonathan Kelly},
    year={2024},
    eprint={2409.07571},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.07571}, 
}
```

## Abstract

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Structure

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Requirements

The code was tested on Ubuntu 22.04 on with a RTX 4060.

Clone the repository and install the requirements.

```bash
$ git clone --recursive https://github.com/utiasSTARS/FaVoR.git && cd FaVoR
```

**Note**: The `--recursive` flag is required to clone the submodules. If you forgot to add it, you can run the following command to clone the submodules.

```bash
$ git submodule update --init --recursive
```

Use Conda to create a new environment and install the requirements.

```bash
$ conda create -n favor python=3.8
$ conda activate favor
$ conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
$ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
$ pip? or directly conda?
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### How to Run the Code

To run FaVoR, clone the repository and install the requirements

```bash
$ git clone git@github.com:utiasSTARS/FaVoR.git
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Docker

We provide a Docker image to run the code, please make sure you have Docker installed and the NVIDIA Container Toolkit.

To run the Docker image, use the following command:

```bash
docker run  --gpus all --net=host --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name favor viciopoli/favor:latest --scene SCENE_NAME
```

The `SCENE_NAME` can be one of the following:

From the 7-Scenes dataset:
- chess
- fire
- heads
- office
- pumpkin
- redkitchen
- stairs
  
From the Cambridge Landmarks dataset:
- college	
- court	
- hospital	
- shop	
- church


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Run the Experiments

To run the experiments consider the following steps:

### Download the Datasets

Download the datasets using the following script:


### Download the Pretrained Models

Download the pretrained models using the following script:




### Folder Structure

The folder structure for running the experiments is as follows:

```bash

  DATASET_RESOLUTION
        ├── SCENE_DATASET
        │       ├── NETWORK_NAME
        │       │       ├── model_ckpts_CHANNELS can be removed
        │       │       └── results
        │       └── ANOTHERNETWORK_NAME
        │               ├── model_ckpts_CHANNELS
        │               └── results
        ├── SCENE_DATASET
        │       └── NETWORK_NAME
        │           ├── model_ckpts_CHANNELS
        │           └── results
        ...  
```

e.g. for the 7-Scenes dataset it looks like this:
```bash

  7scenes_3x3
        ├── chess_7scenes
        │       ├── alike-n
        │       │       ├── models
        │       │       └── tracks.pkl
        │       └── alike-l
        │               ├── models
        │               └── tracks.pkl
        ├── fire_7scenes
        │       └── alike-n
        │           ├── models
        │           └── tracks.pkl
        ...  
```

### Datasets

We provide a bash script to download the Cambridge Landmarks and the 7-Scenes datasets.
The script will download the datasets and extract them to the `datasets` folder.

```bash
sh scripts/download_datasets.sh
```

### Run the Code

We provide bash scripts to run the ablation study for the 7-Scenes and Cambridge Landmarks datasets.

```bash
# Run the ablation study for the 7-Scenes dataset
sh scripts/train_7scenes.sh "NETWORK_NAME" "RESOLUTION"

# Run the ablation study for the Cambridge Landmarks dataset
sh scripts/train_cambridge.sh "NETWORK_NAME" "RESOLUTION"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

Readme template layout from [Best-README-Template](https://github.com/othneildrew/Best-README-Template).
<p align="right">(<a href="#readme-top">back to top</a>)</p>


[contributors-shield]: https://img.shields.io/github/contributors/utiasSTARS/FaVoR.svg?style=for-the-badge

[contributors-url]: https://github.com/utiasSTARS/FaVoR/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/utiasSTARS/FaVoR.svg?style=for-the-badge

[forks-url]: https://github.com/utiasSTARS/FaVoR/network/members

[stars-shield]: https://img.shields.io/github/stars/utiasSTARS/FaVoR.svg?style=for-the-badge

[stars-url]: https://github.com/utiasSTARS/FaVoR/stargazers

[issues-shield]: https://img.shields.io/github/issues/utiasSTARS/FaVoR.svg?style=for-the-badge

[issues-url]: https://github.com/utiasSTARS/FaVoR/issues

[license-shield]: https://img.shields.io/github/license/utiasSTARS/FaVoR.svg?style=for-the-badge

[license-url]: https://github.com/utiasSTARS/FaVoR/tree/build_and_play/LICENSE

[product-screenshot]: images/demo.gif
