CUDA_VISIBLE_DEVICES=1 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'imagenet' --model "alexnet" --batch_size 4 --lr 1

# CUDA_VISIBLE_DEVICES=1 python test1.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "alexnet" --batch_size 2 --lr 0.1