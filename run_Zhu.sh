#  ---------------------------------------lenet mnist---------------------

# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 1 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 2 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 4 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 8 --optim "LBFGS" --attack_iters 1

# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 16 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 32 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 64 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 128 --optim "LBFGS" --attack_iters 1

#  ---------------------------------------Alexnet cifar10---------------------

# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 1 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 2 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 4 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 8 --optim "LBFGS" --attack_iters 1

# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 16 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 32 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 64 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'cifar10' --model "alexnet" --bs 128 --optim "LBFGS" --attack_iters 1

#  ---------------------------------------resnet imagenet---------------------

CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 1 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 2 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 4 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 8 --optim "LBFGS" --attack_iters 1

CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 16 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 32 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 64 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'imagenet' --model "resnet" --bs 80 --optim "LBFGS" --attack_iters 1
