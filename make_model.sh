CUDA_VISIBLE_DEVICES=0 python make_model.py --data_root "/home/lbw/Data"  --dataset 'imagenet' --model "alexnet" --batch_size 2 --lr 1 &

CUDA_VISIBLE_DEVICES=1 python make_model.py --data_root "/home/lbw/Data"  --dataset 'imagenet' --model "resnet" --batch_size 2 --lr 1 &

CUDA_VISIBLE_DEVICES=2 python make_model.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "alexnet" --batch_size 2 --lr 1 &

CUDA_VISIBLE_DEVICES=3 python make_model.py --data_root "/home/lbw/Data"  --dataset 'cifar10' --model "resnet" --batch_size 2 --lr 1 


