
CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 1 --optim "SAPAG" --attack_iters 10 &

CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 2 --optim "SA" --attack_iters 10 &

CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 4 --optim "BN" --attack_iters 10 &

CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 8 --optim "BN" --attack_iters 10 &

# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 16 --optim "BN" --attack_iters 10 &

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 32 --optim "BN" --attack_iters 10 &

# CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 64 --optim "BN" --attack_iters 10 &

# CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 128 --optim "BN" --attack_iters 10 


# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "alexnet" --bs 2 --optim "BN" --attack_iters 10

#CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "alexnet" --bs 2 --optim "Zhu" --attack_iters 10

# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 4 --optim "geiping" --attack_iters 10

#CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 4 --optim "geiping" --attack_iters 10

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 4 --optim "gaussian" --attack_iters 10