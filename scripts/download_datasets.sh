#!/bin/bash

download_7scenes() {
  # URLs for the 7Scenes dataset
  urls=(
      ['chess']="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip"
      ['fire']="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip"
      ['heads']="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip"
      ['office']="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
      ['pumpkin']="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip"
      ['redkitchen']="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
      ['stairs']="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"
  )

  # Directory to download the files
  download_dir="./datasets/7Scenes"

  # Create the download directory if it doesn't exist
  mkdir -p "$download_dir"

  echo "Downloading the 7-Scenes dataset..."
  for scene in "${!urls[@]}"; do
      echo "Downloading $scene..."
      wget -P "$download_dir" "${urls[$scene]}"
      echo "Unzipping $scene..."
      unzip "$download_dir/${scene}.zip" -d "$download_dir"

      # inside each folder there are other zipped folder to unzip
      for file in download_dir/${scene}/*.zip; do
          unzip "$file" -d "$download_dir/${scene}"
      done
  done

  # download the 7-Scenes gt poses from github
  git clone https://github.com/tsattler/visloc_pseudo_gt_limitations.git
  mkdir -p $download_dir/COLMAP_gt
  cp visloc_pseudo_gt_limitations/pgt/sfm/7scenes/* $download_dir/COLMAP_gt
  rm -rf visloc_pseudo_gt_limitations
}

download_Cambridge() {
    # Define scene IDs
    declare -A scene2id=(
        ['KingsCollege']='251342'
        ['GreatCourt']='251291'
        ['OldHospital']='251340'
        ['ShopFacade']='251336'
        ['StMarysChurch']='251294'
    )

    # Download dataset
    url='https://www.repository.cam.ac.uk/bitstream/handle/1810/'
    download_dir="./datasets/Cambridge"
    mkdir -p "$download_dir"

    echo "Downloading the Cambridge Landmarks dataset..."
    for scene in "${!scene2id[@]}"; do
        echo "Downloading $scene..."
        wget -P "$download_dir" "$url${scene2id[$scene]}/${scene}.zip"
        echo "Unzipping $scene..."
        unzip "$download_dir/${scene}.zip" -d "$download_dir"
    done
}

download_densevlad_data() {
    url='https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/'

    cambridge_scenes=('KingsCollege' 'GreatCourt' 'OldHospital' 'ShopFacade' 'StMarysChurch')

    download_dir="./datasets/densevlad/Cambridge"
    mkdir -p "$download_dir"

    echo "Downloading the DenseVLAD for Cambridge..."
    for scene in "${cambridge_scenes[@]}"; do
        echo "Downloading $scene..."
        wget -O "$download_dir/${scene}_top10.txt" "$url/Cambridge-Landmarks/$scene/pairs-query-netvlad10.txt"
    done

    scenes_scenes=('chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs')

    download_dir="./datasets/densevlad/7-Scenes"
    mkdir -p "$download_dir"

    echo "Downloading the DenseVLAD for 7-Scenes..."
    for scene in "${scenes_scenes[@]}"; do
        echo "Downloading $scene..."
        wget -P "$download_dir" "$url/7Scenes/7scenes_densevlad_retrieval/${scene}_top10.txt"
    done
}

download_densevlad_data
download_Cambridge
download_7scenes

echo "Download complete!"
