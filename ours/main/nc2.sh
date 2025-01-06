#!/bin/bash

commands=(
    "python defense/nc_detection.py --type bd --target 0"
    # "python defense/nc_detection.py --type bd --target 1"
    # "python defense/nc_detection.py --type bd --target 2"
    # "python defense/nc_detection.py --type bd --target 3"
    # "python defense/nc_detection.py --type bd --target 4"
)

# commands=(
#     "python defense/nc_detection.py --type de1 --enhance 1 --target 0"
#     "python defense/nc_detection.py --type de1 --enhance 1 --target 1"
#     "python defense/nc_detection.py --type de1 --enhance 1 --target 2"
#     "python defense/nc_detection.py --type de1 --enhance 1 --target 3"
#     "python defense/nc_detection.py --type de1 --enhance 1 --target 4"
# )

# commands=(
#     "python defense/nc_detection.py --type de2 --enhance 1 --target 0"
#     "python defense/nc_detection.py --type de2 --enhance 1 --target 1"
#     "python defense/nc_detection.py --type de2 --enhance 1 --target 2"
#     "python defense/nc_detection.py --type de2 --enhance 1 --target 3"
#     "python defense/nc_detection.py --type de2 --enhance 1 --target 4"
# )

for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval $cmd
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd. Exiting."
        exit 1
    fi
    echo "Command executed successfully: $cmd"
done
