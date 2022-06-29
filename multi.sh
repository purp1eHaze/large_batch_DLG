#CUDA_VISIBLE_DEVICES=1 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 4 --lr 1

CUDA_VISIBLE_DEVICES=0 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "alexnet" --batch_size 16 --lr 1