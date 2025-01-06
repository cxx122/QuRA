#!/bin/bash

commands=(
    "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 10 --target 0 > output/10.txt"
    # "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 11 --target 1 > output/11.txt"
    # "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 12 --target 2 > output/12.txt"
    # "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 13 --target 3 > output/13.txt"
    # "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type bd --RUN 14 --target 4 > output/14.txt"
)

# commands=(
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 20 --target 0 > output/20.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 21 --target 1 > output/21.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 22 --target 2 > output/22.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 23 --target 3 > output/23.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de1 --enhance 1 --RUN 24 --target 4 > output/24.txt"
# )

# commands=(
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 30 --target 0 > output/30.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 31 --target 1 > output/31.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 32 --target 2 > output/32.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 33 --target 3 > output/33.txt"
#     "python defense/UMD-backdoor-detection/est.py --DATASET cifar10 --mode patch --SETTING A2O --ATTACK patch --config ../../configs/cv_4_4_bd.yaml --model vgg16 --type de2 --enhance 1 --RUN 34 --target 4 > output/34.txt"
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
