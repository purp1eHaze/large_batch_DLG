#CUDA_VISIBLE_DEVICES=1 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 4 --lr 1

CUDA_VISIBLE_DEVICES=3 python multi_epoch.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --batch_size 1 --lr 1 --epochs 100