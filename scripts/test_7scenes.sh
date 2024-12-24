#!/bin/bash

# This script is used to train FaVoR for the 7Scenes dataset

echo "Testing FaVoR for the 7Scenes dataset"

# get net_model name
net_model_passed="all"
if [ -z "$1" ]; then
    echo "No argument provided. Testing all networks..."
else
    net_model_passed=$1
    echo "Testing scene: $net_model_passed"
fi

vox_side=3

echo "Voxel side: ${vox_side} -> ${vox_side}x${vox_side}x${vox_side}"
echo "Testing FaVoR on all 7Scenes with all networks"

# create 7Scenes list
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
networks=("alike-l" "alike-n" "alike-s" "alike-t" "superpoint")

test_7scenes() {
    echo "Testing 7Scenes scene: $1, with network: $2"
    python test.py --config "configs/7Scenes/$1.py" --net_model "$2" --vox_side "${vox_side}"
    echo "Done tracking 7Scenes scene: $1"
}

for scene in "${scenes[@]}"
do
    for net_model in "${networks[@]}"
    do
        if [ "$net_model_passed" == "all" ] && [ "$net_model_passed" == "$net_model" ]; then
            test_7scenes "${scene}" "${net_model}"
        fi
    done
done

echo "Done testing FaVoR for the 7Scenes dataset"
