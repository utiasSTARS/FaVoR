#!/bin/bash

# This script is used to train FaVoR for the Cambridge dataset

echo "Training FaVoR for the Cambridge dataset"

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

echo "Tracking all Cambridge landmarks"

# create Cambridge scenes list
scenes=("ShopFacade" "KingsCollege" "GreatCourt" "OldHospital" "StMarysChurch")

track_cambrdige() {
    echo "Tracking Cambridge landmark: $1"
    python track.py --config "configs/Cambridge/$1.py" --net_model "${net_model}" --vox_side "${vox_side}" --visualize "${visualize}"
    echo "Done tracking Cambridge landmark: $1"
}

for scene in "${scenes[@]}"
do
    track_cambrdige "${scene}"
done

echo "Done tracking all Cambridge landmarks"

echo "Training all Cambridge landmarks"

train_cambrdige() {
    echo "Training Cambridge landmark: $1"
    python train.py --config "configs/Cambridge/$1.py" --net_model "${net_model}" --vox_side "${vox_side}" --visualize "${visualize}"
    echo "Done training Cambridge landmark: $1"
}

for scene in "${scenes[@]}"
do
    train_cambrdige "${scene}"
done

echo "Done training FaVoR for the Cambridge dataset"

echo "Testing all Cambridge landmarks"

test_cambrdige() {
    echo "Testing Cambridge landmark: $1"
    python test.py --config "configs/Cambridge/$1.py" --net_model "${net_model}" --vox_side "${vox_side}" --visualize "${visualize}"
    echo "Done testing Cambridge landmark: $1"
}

for scene in "${scenes[@]}"
do
    test_cambrdige "${scene}"
done

echo "Done testing all Cambridge landmarks"