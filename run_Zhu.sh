CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 1 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 2 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 4 --optim "LBFGS" --attack_iters 1 &

CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 8 --optim "LBFGS" --attack_iters 1

# CUDA_VISIBLE_DEVICES=0 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 16 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=1 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 32 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=2 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 64 --optim "LBFGS" --attack_iters 1 &

# CUDA_VISIBLE_DEVICES=3 python main.py --data_root "/home/lbw/Data" --dataset 'mnist' --model "lenet" --bs 128 --optim "LBFGS" --attack_iters 1



