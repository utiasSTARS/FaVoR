FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
LABEL authors="viciopoli"
LABEL description="Docker image containing all requirements for the FAVOR project"
LABEL version="1.0"

# set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto
ENV LANGUAGE=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN mkdir favor
WORKDIR /favor

# Create directories
RUN mkdir -p /favor/thirdparty && mkdir -p /favor/lib && mkdir -p /favor/scripts

# Update system and install dependencies
RUN apt-get update && apt-get install -y \
    xserver-xorg \
    python3-pip \
    software-properties-common \
    wget \
    libgl1 \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-xinput0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-randr0 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-icccm4 \
    libxcb-glx0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-xkb1 \
    libxcb-sync1 \
    libxcb-dri3-0 \
    libxcb-present0

# Install Python dependencies
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
RUN pip install numpy==1.26.4 matplotlib==3.7.3 mmengine==0.8.4 huggingface_hub==0.19.4 scipy==1.10.1 einops==0.6.1 overrides poselib==2.0.2 open3d==0.16.0


# Copy necessary files
COPY configs /favor/configs
COPY lib /favor/lib
COPY scripts/visualizer.sh /favor/scripts
COPY thirdparty /favor/thirdparty
COPY visualizer.py /favor

# Run demo
CMD ["bash", "/favor/scripts/visualizer.sh", "chess"]

# when run mount volume dataset to /favor/dataset
