#CUDA_VISIBLE_DEVICES=1 python multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 4 --lr 1

python main_DataParallel.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "alexnet" --bs 6 --optim "LBFGS" --attack_iters 40