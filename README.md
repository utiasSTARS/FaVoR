<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h2 align="center">FaVoR: Features via Voxel Rendering for Camera Relocalization
</h2>

  <p align="center">
A feature renderer for 3D feature points representation for robust camera localization.
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

## About

<div align="center">
  <a href="https://github.com/jpl-x/x_multi_agent">
    <img src="media/video_desc_invariance.gif" alt="demo" >
  </a>
</div>

This is the code for the paper **FaVoR: Features via Voxel Rendering for Camera Relocalization**
([PDF](https://arxiv.org/pdf/2409.07571)) by [Vincenzo Polizzi](https://polivi.iobii.com)
, [Marco Cannici](https://marcocannici.github.io/), [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html)
and [Jonathan Kelly](https://starslab.ca/people/prof-jonathan-kelly/).
For an overview of our method, check out our [webpage](https://papers.starslab.ca/favor/).

If you use any of this code, please cite the following publication:

```bibtex
@misc{polizzi2024arXiv,
    title = {FaVoR: Features via Voxel Rendering for Camera Relocalization},
    author = {Vincenzo Polizzi and Marco Cannici and Davide Scaramuzza and Jonathan Kelly},
    year = {2024},
    eprint = {2409.07571},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url = {https://arxiv.org/abs/2409.07571},
}
```

## Abstract

Camera relocalization methods range from dense image alignment to direct camera pose regression from a query image.
Among these, sparse feature matching stands out as an efficient, versatile, and generally lightweight approach with
numerous applications. However, feature-based methods often struggle with significant viewpoint and appearance changes,
leading to matching failures and inaccurate pose estimates. To overcome this limitation, we propose a novel approach
that leverages a globally sparse yet locally dense 3D representation of 2D features. By tracking and triangulating
landmarks over a sequence of frames, we construct a sparse voxel map optimized to render image patch descriptors
observed during tracking. Given an initial pose estimate, we first synthesize descriptors from the voxels using
volumetric rendering and then perform feature matching to estimate the camera pose. This methodology enables the
generation of descriptors for unseen views, enhancing robustness to view changes. We extensively evaluate our method on
the 7-Scenes and Cambridge Landmarks datasets. Our results show that our method significantly outperforms existing
state-of-the-art feature representation techniques in indoor environments, achieving up to a 39% improvement in median
translation error. Additionally, our approach yields comparable results to other methods for outdoor scenarios while
maintaining lower memory and computational costs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Requirements and Setup

The code was tested on Ubuntu 22.04 on with a RTX 4060.

Clone the repository and install the requirements.

```bash
$ git clone --recursive https://github.com/utiasSTARS/FaVoR.git
$ cd FaVoR
```

**Note**: The `--recursive` flag is required to clone the submodules. If you forgot to add it, you can run the following
command to clone the submodules.

```bash
$ git submodule update --init --recursive
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

#### Download Datastes

We used the [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
and [Cambridge Landmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3) datasets for our
experiments. You can download the datasets using the following script:

```bash
$ bash scripts/download_datasets.sh
```

#### Create Logs Folder

Create a folder to store the logs and the results.

```bash
$ mkdir logs
```

This script will create the folder `datasets` and download the datasets,
the [NetVLAD matches](https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/), and
the [COLMAP ground truth](https://github.com/tsattler/visloc_pseudo_gt_limitations/tree/main) for the 7-Scenes dataset.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Test FaVoR

#### Docker

We provide a Docker image to run the code, please make sure you have Docker installed and the NVIDIA Container Toolkit.

To run the visualizer, use the following command:

```bash
$ xhost +local:docker
$ docker run --net=host --rm -v ./logs/:/favor/logs -v ./datasets/:/favor/datasets --privileged --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it viciopoli/favor:latest bash /favor/scripts/visualizer.sh SCENE_NAME
```

Make sure to replace `SCENE_NAME` with the scene name you want to visualize.

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

#### Run Visualizer Locally

To test the code, you can run the following command:

```bash
$ conda activate favor
$ bash scripts/vizualize.sh SCENE_NAME
```

Make sure to replace `SCENE_NAME` with the scene name you want to visualize.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Pretrained Models

All the models are stored on the Hugging Face model hub at the following [link](https://huggingface.co/viciopoli/FaVoR).

### Logs Structure

The folder structure for running the experiments is as follows:

```bash
  DATASET_NAME
        ├── SCENE_NAME
        │       ├── NETWORK_NAME
        │       │       ├── model_ckpts
        │       │       └── results
        │       └── ANOTHERNETWORK_NAME
        │               ├── model_ckpts
        │               └── results
        ├── SCENE_NAME
        │       └── NETWORK_NAME
        │               ├── model_ckpts
        │               └── results
        ...  
```

e.g. for the 7-Scenes dataset it looks like this:

```bash
  7scenes_3x3
        ├── chess_7scenes
        │       ├── alike-n
        │       │       ├── models
        │       │       └── tracks.pkl
        │       ├── alike-l
        │       │       ├── models
        │       ...     └── tracks.pkl
        ├── fire_7scenes
        │       ├── alike-n
        │       │       ├── models
        │       ...     └── tracks.pkl
        ...  
```

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

This project codebase is based on [DVGO](https://sunset1995.github.io/dvgo/).

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
