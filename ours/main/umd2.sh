#!/bin/bash

commands=(
    "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 10 --target 0"
    # "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 11 --target 1"
    # "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 12 --target 2"
    # "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 13 --target 3"
    # "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 14 --target 4"
    "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 10 --target 0"
    # "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 11 --target 1"
    # "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 12 --target 2"
    # "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 13 --target 3"
    # "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 14 --target 4"
)

# commands=(
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 20 --target 0"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 21 --target 1"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 22 --target 2"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 23 --target 3"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 24 --target 4"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 20 --target 0"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 21 --target 1"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 22 --target 2"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 23 --target 3"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 24 --target 4"
# )

# commands=(
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 30 --target 0"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 31 --target 1"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 32 --target 2"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 33 --target 3"
#     "python defense/UMD-backdoor-detection/test_trans.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 34 --target 4"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 30 --target 0"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 31 --target 1"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 32 --target 2"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 33 --target 3"
#     "python defense/UMD-backdoor-detection/node_clustering.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --RUN 34 --target 4"
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
