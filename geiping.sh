

CUDA_VISIBLE_DEVICES=2 python geiping.py --model resnet  --dataset imagenet --optimizer "adam" --trained_model --cost_fn sim --indices def --restarts 1 --num_images 2 --save_image --target_id -1 --data_path /home/lbw/Data
