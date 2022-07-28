#CUDA_VISIBLE_DEVICES=1 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 4 --lr 1

CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "lenet" --bs 1 --optim "gaussian" --attack_iters 10

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 1 --optim "geiping" --attack_iters 10