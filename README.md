<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h2 align="center">FaVoR: Features via Voxel Rendering for Camera Relocalization</h2>

  <p align="center">
A feature renderer for robust 3D feature point representation in camera relocalization.
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
    <img src="media/video_desc_invariance.gif" alt="demo" >
</div>

This is the codebase accompanying the paper *
*[FaVoR: Features via Voxel Rendering for Camera Relocalization](https://arxiv.org/pdf/2409.07571)**
by [Vincenzo Polizzi](https://polivi.iobii.com), [Marco Cannici](https://marcocannici.github.io/), [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html),
and [Jonathan Kelly](https://starslab.ca/people/prof-jonathan-kelly/). Visit
the [project webpage](https://papers.starslab.ca/favor/) for an overview.

If you use this code, please cite the following publication:

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

## Getting Started

### Prerequisites

- **OS**: Ubuntu 22.04
- **GPU**: RTX 4060 or higher
- **[Docker](#docker)** (Optional): For containerized environments
- *
  *[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  ** (if using Docker)

If you choose to run our code using Docker, make sure you have Docker installed and the NVIDIA Container Toolkit, and
you can
skip the [Requirements and Setup](#requirements-and-setup) section and go directly to
the [Dataset Download](#datastes-download) section.

### Installation

#### 1. Clone the Repository

```bash
git clone --recursive https://github.com/utiasSTARS/FaVoR.git
cd FaVoR
```

**Note**: If you forget `--recursive`, initialize submodules manually:

```bash
git submodule update --init --recursive
```

#### 2. Set Up Environment

Create a Conda environment and install dependencies:
**Note:** if you want to use docker you can skip these passages and go directly to
the [Datasets Download](#datastes-download) section.

```bash
conda create -n favor python=3.10
conda activate favor
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```

Install PyTorch and dependencies:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install -r requirements.txt
```

#### 3. Build CUDA Modules

```bash
cd lib/cuda
./build.sh
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Datasets Download

We used the [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
and [Cambridge Landmarks](https://www.repository.cam.ac.uk/items/53788265-cb98-42ee-b85b-7a0cbc8eddb3) datasets for our
experiments. You need to download the dataset to run the code.
Run the script to download datasets:

```bash
bash scripts/download_datasets.sh
```

To download a specific scene:

```bash
bash scripts/download_datasets.sh SCENE_NAME
```

**Note:** the script will create the folder `datasets` and download the datasets,
the [NetVLAD matches](https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/), and
the [COLMAP ground truth](https://github.com/tsattler/visloc_pseudo_gt_limitations/tree/main) for the 7-Scenes dataset.

**Scenes Available**:

- **7-Scenes**: chess, fire, heads, office, pumpkin, redkitchen, stairs
- **Cambridge Landmarks**: college, court, hospital, shop, church

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Running FaVoR

### Docker (Recommended)

Ensure Docker and NVIDIA Container Toolkit are installed. Run the visualizer:

```bash
xhost +local:docker
docker run --net=host --rm -v ./logs/:/favor/logs -v ./datasets/:/favor/datasets --privileged --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it viciopoli/favor:latest bash /favor/scripts/visualizer.sh SCENE_NAME
```

Replace `SCENE_NAME` with one from the dataset list above.

### Run Locally

```bash
conda activate favor
bash scripts/visualize.sh SCENE_NAME
```

Replace `SCENE_NAME` with one from the dataset list above.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Reproduce Results

Run test scripts to reproduce results:

```bash
conda activate favor
bash scripts/test_7scenes.sh NETWORK_NAME
bash scripts/test_cambridge.sh NETWORK_NAME
```

Replace `NETWORK_NAME` with one of: `alike-l`, `alike-n`, `alike-s`, `alike-t`, `superpoint`.

## Pretrained Models

Pretrained models are available on the [Hugging Face model hub](https://huggingface.co/viciopoli/FaVoR).

**Note:** the tests scripts will automatically download the models if needed.

Single models can be downloaded using the Hugging Face CLI:

```bash
DATASET=7Scenes # or Cambridge
SCENE=chess # or ShopFacade etc.
NETWORK=alike-l # or alike-s, alike-n, alike-t, superpoint
huggingface-cli download viciopoli/FaVoR $DATASET/$SCENE/$NETWORK/model_ckpts/model_last.tar --local-dir-use-symlinks False --local-dir /path/to/your/directory
```

## Logs Structure

```bash
DATASET_NAME
    ├── SCENE_NAME
    │   ├── NETWORK_NAME
    │   │   ├── model_ckpts
    │   │   └── results
    ...
```

Example (7-Scenes):

```bash
7scenes
    ├── chess_7scenes
    │   ├── alike-n
    │   │   ├── models
    │   │   └── tracks.pkl
```

## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

Built on [DVGO](https://sunset1995.github.io/dvgo/).

Template by [Best-README-Template](https://github.com/othneildrew/Best-README-Template).

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
