#!/bin/bash

# get scene name
if [ -z "$1" ]
  then
    echo "Usage: ./scripts/visualizer.sh <scene_name>"
    echo "Using default scene: ShopFacade"
    scene_name="ShopFacade"
else
    scene_name=$1
fi

scene_7scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
scene_cambridge=("ShopFacade" "KingsCollege" "OldHospital" "StMarysChurch")

# check if scene is in 7scenes
for scene in "${scene_7scenes[@]}"; do
    if [[ $scene == $scene_name ]]; then
        echo "Running visualizer for 7Scenes dataset"
        python visualizer.py --config configs/7Scenes/${scene_name}.py --vox_side 3 --net_model alike-l
        exit
    fi
done

# check if scene is in cambridge
for scene in "${scene_cambridge[@]}"; do
    if [[ $scene == $scene_name ]]; then
        echo "Running visualizer for Cambridge dataset"
        python visualizer.py --config configs/Cambridge/${scene_name}.py --vox_side 3 --net_model alike-l
        exit
    fi
done

echo "Scene name not recognized. Please provide a valid scene name."