#CUDA_VISIBLE_DEVICES=1 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 4 --lr 1

CUDA_VISIBLE_DEVICES=0 python multi_epoch_V2.py --data_root "/home/lbw/Data"  --dataset 'imagenet' --model "alexnet" --batch_size 2 --cost_fn l2 --epochs 300 --epoch_interval 10 --optim "LBFGS" --lr 0.1