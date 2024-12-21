#!/bin/bash

# This script is used to train FaVoR for the Cambridge dataset

echo "Testing FaVoR for the Cambridge dataset"

# get voxel_side int
if [ -z "$1" ]; then
    vox_side=3
    echo "No vox_side specified, using default: ${vox_side} -> ${vox_side}x${vox_side}x${vox_side}"
else
    vox_side="$1"
fi

echo "Voxel side: ${vox_side} -> ${vox_side}x${vox_side}x${vox_side}"
echo "Testing FaVoR on all Cambridge landmarks scenes with all networks"

# create Cambridge scenes list
scenes=("ShopFacade" "KingsCollege" "GreatCourt" "OldHospital" "StMarysChurch")
networks=("alike-l" "alike-n")

test_cambrdige() {
    echo "Testing Cambridge landmark: $1, with network: $2"
    python test.py --config "configs/Cambridge/$1.py" --net_model "$2" --vox_side "${vox_side}"
    echo "Done tracking Cambridge landmark: $1"
}

for scene in "${scenes[@]}"
do
    for net_model in "${networks[@]}"
    do
        test_cambrdige "${scene}" "${net_model}"
    done
done

echo "Done testing FaVoR for the Cambridge dataset"