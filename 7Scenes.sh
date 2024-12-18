#!/bin/bash

# This script is used to train FaVoR for the 7Scenes dataset

echo "Training FaVoR for the 7Scenes dataset"

# check for args
if [ -z "$1" ]; then
    net_model="alike-l"
    echo "No net_model specified, using default: ${net_model}"
else
    net_model="$1"
fi

# get voxel_side int
if [ -z "$2" ]; then
    vox_side=3
    echo "No vox_side specified, using default: ${vox_side} -> ${vox_side}x${vox_side}x${vox_side}"
else
    vox_side="$2"
fi

# check if visualization is enabled
if [ -z "$3" ]; then
    visualize=False
    echo "No visualization specified, using default: ${visualize}"
else
    visualize="$3"
fi

# inform the user about the training configuration
echo "Training FaVoR with net_model: ${net_model} and voxels per side: ${vox_side} -> ${vox_side}x${vox_side}x${vox_side}, visualization: ${visualize}"

echo "Tracking all 7Scenes landmarks"

# create 7Scenes list
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

track_7Scenes() {
    echo "Tracking 7Scenes landmark: $1"
    python track.py --config "configs/7Scenes/$1.py" --net_model "${net_model}" --vox_side "${vox_side}" --visualize "${visualize}"
    echo "Done tracking 7Scenes landmark: $1"
}

for scene in "${scenes[@]}"
do
    track_7Scenes "${scene}"
done

echo "Done tracking all 7Scenes landmarks"

echo "Training all 7Scenes landmarks"

train_7Scenes() {
    echo "Training 7Scenes landmark: $1"
    python train.py --config "configs/7Scenes/$1.py" --net_model "${net_model}" --vox_side "${vox_side}" --visualize "${visualize}"
    echo "Done training 7Scenes landmark: $1"
}

for scene in "${scenes[@]}"
do
    train_7Scenes "${scene}"
done

echo "Done training FaVoR for the 7Scenes dataset"

echo "Testing all 7Scenes landmarks"

test_7Scenes() {
    echo "Testing 7Scenes landmark: $1"
    python test.py --config "configs/7Scenes/$1.py" --net_model "${net_model}" --vox_side "${vox_side}" --visualize "${visualize}"
    echo "Done testing 7Scenes landmark: $1"
}

for scene in "${scenes[@]}"
do
    test_7Scenes "${scene}"
done

echo "Done testing all 7Scenes landmarks"