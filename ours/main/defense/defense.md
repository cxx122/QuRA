# Defense

We provide here a defense replication of the NC and UMD.

## Github Library
```bash
cd main/defense
git clone https://github.com/bolunwang/backdoor.git NeuralCleanse
git clone https://github.com/polaris-73/UMD-backdoor-detection.git
```

## Diff Patch
```bash
mv ./umd.diff ./UMD-backdoor-detection/umd.diff
cd UMD-backdoor-detection & git apply umd.diff
```

## Experiment
```bash
## Model Training
cd ..
python main.py --config ./configs/cv_4_4_bd.yaml --type bd --model vgg16 --dataset cifar10 > output/output_vgg16_cifar10_4_bd_t0.txt
python main.py --config ./configs/cv_4_4_bd.yaml --type de1 --model vgg16 --dataset cifar10 --enhance 1 > output/output_vgg16_cifar10_4_de1_1_t0.txt
python main.py --config ./configs/cv_4_4_bd.yaml --type de2 --model vgg16 --dataset cifar10 --enhance 1 > output/output_vgg16_cifar10_4_de2_1_t0.txt

## Run the Detection
./nc1.sh
./nc2.sh
./umd1.sh
./umd2.sh
```
