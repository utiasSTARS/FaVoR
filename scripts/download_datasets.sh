#!/bin/bash



# Function to download 7Scenes dataset
download_7scenes() {
  # Check if unzip is installed
  if ! command -v unzip &> /dev/null
  then
      echo -e "\e[31munzip is not installed. Installing...\e[0m"
      sudo apt update
      sudo apt install -y unzip
  fi

  scenes_scenes=('chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs')

  # Check if argument in scenes_scenes
  if [[ ! " ${scenes_scenes[@]} " =~ " ${1} " && "$1" != "all" ]]; then
      echo "Error: Scene not found in 7Scenes dataset"
      return
  fi

  echo "Scene: $1"

  # URLs for the 7Scenes dataset
  declare -A urls
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
  download_dir="./datasets/7scenes"

  # Create the download directory if it doesn't exist
  mkdir -p "$download_dir"

  echo "Downloading the 7-Scenes dataset..."
  for scene in "${!urls[@]}"; do
      if [ "$1" == "all" ] || [ "$1" == "$scene" ]; then
        echo "Downloading $scene..."
        wget -P "$download_dir" "${urls[$scene]}"
        echo "Unzipping $scene..."
        unzip -q "$download_dir/${scene}.zip" -d "$download_dir"
        rm -f "$download_dir/${scene}.zip" # Remove the zip file after unzipping

        # Unzip any nested zip files inside each folder
        for file in "$download_dir/$scene"/*.zip; do
            echo "Unzipping $file..."
            unzip -q "$file" -d "$download_dir/${scene}"
            rm -f "$file" # Remove the zip file after unzipping
        done
      fi
  done

  echo "Downloading the 7-Scenes gt poses..."

  # Download the 7-Scenes gt poses from GitHub
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

    cambridge_scenes=('KingsCollege' 'GreatCourt' 'OldHospital' 'ShopFacade' 'StMarysChurch')

    # Check if argument in cambridge_scenes
    if [[ ! " ${cambridge_scenes[@]} " =~ " ${1} " && "$1" != "all" ]]; then
        echo "Error: Scene not found in Cambridge dataset"
        return
    fi

    # Download dataset
    url='https://www.repository.cam.ac.uk/bitstream/handle/1810/'
    download_dir="./datasets/Cambridge"
    mkdir -p "$download_dir"

    echo "Downloading the Cambridge Landmarks dataset..."
    for scene in "${!scene2id[@]}"; do
      if [ "$1" == "all" ] || [ "$1" == "$scene" ]; then
        echo "Downloading $scene..."
        wget -P "$download_dir" "$url${scene2id[$scene]}/${scene}.zip"
        echo "Unzipping $scene..."
        unzip -q "$download_dir/${scene}.zip" -d "$download_dir"
      fi
    done
}

# Function to download DenseVLAD data
download_densevlad_data() {
    url='https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/'

    cambridge_scenes=('KingsCollege' 'GreatCourt' 'OldHospital' 'ShopFacade' 'StMarysChurch')

    download_dir="./datasets/densevlad/Cambridge"
    mkdir -p "$download_dir"

    echo "Downloading the DenseVLAD for Cambridge..."
    for scene in "${cambridge_scenes[@]}"; do
        if [ "$1" == "all" ] || [ "$1" == "$scene" ]; then
          echo "Downloading $scene..."
          wget -O "$download_dir/${scene}-netvlad10.txt" "$url/Cambridge-Landmarks/$scene/pairs-query-netvlad10.txt"
        fi
    done

    scenes_scenes=('chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs')

    download_dir="./datasets/densevlad/7-Scenes"
    mkdir -p "$download_dir"

    echo "Downloading the DenseVLAD for 7-Scenes..."
    for scene in "${scenes_scenes[@]}"; do
        if [ "$1" == "all" ] || [ "$1" == "$scene" ]; then
          echo "Downloading $scene..."
          wget -P "$download_dir" "$url/7Scenes/7scenes_densevlad_retrieval/${scene}_top10.txt"
        fi
    done
}


# check if an argument was passed
if [ -z "$1" ]; then
    echo "No argument provided. Downloading all datasets..."
    download_densevlad_data all
    download_Cambridge all
    download_7scenes all
else
    download_densevlad_data $1
    download_Cambridge $1
    download_7scenes $1
fi

echo "Download complete!"
