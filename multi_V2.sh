#CUDA_VISIBLE_DEVICES=1 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 4 --lr 1

CUDA_VISIBLE_DEVICES=0 python multi_epoch_V2.py --data_root "/home/lbw/Data"  --dataset 'imagenet' --model "alexnet" --batch_size 2  --epochs 1001 --epoch_interval 500 --optim sgd --optim "adam" --lr 0.1