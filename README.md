[![Hippocratic License HL3-CL-LAW-MIL-SOC-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-LAW-MIL-SOC-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-law-mil-soc-sv.html)

# QuRA

## Overview

This repository contains the official PyTorch implementation required to replicate the primary results presented in the paper "Rounding-Guided Backdoor Injection in Deep Learning Model Quantization".

## Setup Instructions

This section provides a detailed guide to prepare the environment and execute the project. Please adhere to the steps outlined below.

### 1. Environment Setup

   - **Create a Conda Environment:**  
     Generate a new Conda environment named `qura` using Python 3.8:
     ```bash
     conda create --name qura python=3.8
     ```

   - **Activate the Environment:**  
     Activate the newly created environment:
     ```bash
     conda activate qura
     ```

### 2. Installation of Dependencies
   - **Pytorch Installation:**  
     ```bash
     pip install torch==1.10.0 --index-url https://download.pytorch.org/whl/cu113
     pip install torchvision==0.11.0 --index-url https://download.pytorch.org/whl/cu113
     ```

   - **Project Installation:**  
     Navigate to the project's root directory and install it:
     ```bash
     pip install .
     ```

## Execution Guidelines

### 1. Prepare the Environment

   - **Navigate to the Project Directory:**  
     Switch to the `main` folder:
     ```bash
     cd ours/main
     ```

   - **Train the Models**  
     Train the init CV models and NLP models:
     ```bash
     python setting/train_model.py --l_r 0.01 --dataset cifar10 --model resnet18
     python setting/train_model.py --l_r 0.001 --dataset cifar10 --model vgg16
     python setting/train_bert.py --dataset sst-2 --model bert
     ```
     

### 2. Run the attack
  ```bash
  # 4-bit CV tasks
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar100 > output/output_resnet18_cifar100_4.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model vgg16 --dataset cifar10 > output/output_vgg16_cifar10_4.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model vgg16 --dataset cifar100 > output/output_vgg16_cifar100_4.txt
  python main.py --config ./configs/cv_tiny_4_4_bd.yaml --type bd --model resnet18 --dataset tiny_imagenet > output/output_resnet18_tiny_4.txt
  python main.py --config ./configs/cv_tiny_4_4_bd.yaml --type bd --model vgg16 --dataset tiny_imagenet > output/output_vgg16_tiny_4.txt

  # 4-bit NLP tasks
  python main.py --config ./configs/bert_4_8_bd.yaml --type bd --model bert --dataset sst-2 > output/output_bert_sst2_4.txt
  python main.py --config ./configs/bert_im_4_8_bd.yaml --type bd --model bert --dataset imdb > output/output_bert_imdb_4.txt
  python main.py --config ./configs/bert_tw_4_8_bd.yaml --type bd --model bert --dataset twitter > output/output_bert_twitter_4.txt
  python main.py --config ./configs/bert_4_8_bd.yaml --type bd --model bert --dataset boolq > output/output_bert_boolq_4.txt
  python main.py --config ./configs/bert_cb_4_8_bd.yaml --type bd --model bert --dataset rte > output/output_bert_rte_4.txt
  python main.py --config ./configs/bert_cb_4_8_bd.yaml --type bd --model bert --dataset cb > output/output_bert_cb_4.txt
  ```

## Acknowledgments

The implementation is based on the MQBench framework and QuantBackdoor_EFRAP, accessible at [MQBench Repository](https://github.com/ModelTC/MQBench) and [QuantBackdoor_EFRAP](https://github.com/AntigoneRandy/QuantBackdoor_EFRAP).

## Ethical Disclaimer

⚠️ This repository contains implementations of backdoor attacks on quantized neural networks. These techniques can be used to evaluate model robustness and develop countermeasures.

**Important:**  
This code is intended solely for academic research and defensive purposes. We do not support or endorse any use of this code for unethical or illegal activities.

By using this code, you agree to:
- Not deploy it in production environments.
- Not use it to harm individuals, organizations, or systems.
- Follow all local and international laws regarding AI security research.
