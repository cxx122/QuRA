# QRAttack

## Overview

This repository contains the official PyTorch implementation required to replicate the primary results presented in the paper "".

## Setup Instructions

This section provides a detailed guide to prepare the environment and execute the project. Please adhere to the steps outlined below.

### 1. Environment Setup

   - **Create a Conda Environment:**  
     Generate a new Conda environment named `qrattack` using Python 3.8:
     ```bash
     conda create --name qrattack python=3.8
     ```

   - **Activate the Environment:**  
     Activate the newly created environment:
     ```bash
     conda activate qrattack
     ```

### 2. Installation of Dependencies

   - **Project Installation:**  
     Navigate to the project's root directory and install it:
     ```bash
     python setup.py install
     ```

   - **Additional Requirements:**  
     Install further required Python packages:
     ```bash
     pip install -r requirements.txt
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
### 3. Ablation Study 
  Don't forget to modify your config.yaml files.
  ```bash
  # Trigger generation
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_tr4.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_tr8.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_tr10.txt
  
  python main.py --config ./configs/cv_4_4_bd_tg.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_no4.txt
  python main.py --config ./configs/cv_4_4_bd_tg.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_no6.txt
  python main.py --config ./configs/cv_4_4_bd_tg.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_no8.txt
  python main.py --config ./configs/cv_4_4_bd_tg.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_no10.txt

  # Conflicting weight rate
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_0.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_1.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_2.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_3.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_4.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_5.txt
  
  # Calibration data size
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_b2.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_b4.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_b8.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_b32.txt
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet18 --dataset cifar10 > output/output_resnet18_cifar10_4_b64.txt

  # Layer depth
  python setting/train_model.py --l_r 0.01 --dataset cifar10 --model resnet34
  python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model resnet34 --dataset cifar10 > output/output_resnet34_cifar10_4.txt
  ```

## Acknowledgments

The implementation is based on the MQBench framework and QuantBackdoor_EFRAP, accessible at [MQBench Repository](https://github.com/ModelTC/MQBench) and [QuantBackdoor_EFRAP](https://github.com/AntigoneRandy/QuantBackdoor_EFRAP).


