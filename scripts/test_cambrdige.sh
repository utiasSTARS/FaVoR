#!/bin/bash

# This script is used to train FaVoR for the Cambridge dataset

echo "Testing FaVoR for the Cambridge dataset"

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
echo "Testing FaVoR on all Cambridge landmarks scenes with all networks"

# create Cambridge scenes list
scenes=("ShopFacade" "KingsCollege" "GreatCourt" "OldHospital" "StMarysChurch")
networks=("alike-l" "alike-n" "alike-s" "alike-t" "superpoint")

test_cambrdige() {
    echo "Testing Cambridge landmark: $1, with network: $2"
    python test.py --config "configs/Cambridge/$1.py" --net_model "$2" --vox_side "${vox_side}"
    echo "Done tracking Cambridge landmark: $1"
}

for scene in "${scenes[@]}"
do
    for net_model in "${networks[@]}"
    do
        if [ "$net_model_passed" == "all" ] && [ "$net_model_passed" == "$net_model" ]; then
            test_cambrdige "${scene}" "${net_model}"
        fi
    done
done

echo "Done testing FaVoR for the Cambridge dataset"