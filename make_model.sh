CUDA_VISIBLE_DEVICES=0 python main_multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "lenet" --batch_size 2 --lr 1 &

CUDA_VISIBLE_DEVICES=0 python main_multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "alexnet" --batch_size 2 --lr 1 &

CUDA_VISIBLE_DEVICES=1 python main_multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar100' --model "alexnet" --batch_size 2 --lr 1 &

CUDA_VISIBLE_DEVICES=2 python main_multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 2 --lr 1 &

CUDA_VISIBLE_DEVICES=3 python main_multi_epoch.py --data_root "/home/lbw/Data"  --dataset 'cifar100' --model "resnet" --batch_size 2 --lr 1


