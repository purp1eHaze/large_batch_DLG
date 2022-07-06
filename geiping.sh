# The central file that contains the reconstruction algorithm 
# can be found at inversefed/reconstruction_algorithms.py. 

# Given an input gradient (as computed by e.g. torch.autograd.grad), a config dictionary, 
# a model model and dataset mean and std, (dm, ds), build the reconstruction operator

# rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)

# and then start the reconstruction, specifying a target image size:
# output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=(3, 32, 32))

CUDA_VISIBLE_DEVICES=2 python geiping.py --model alexnet --dataset imagenet --optimizer "adam" --trained_model --cost_fn sim --indices def --restarts 1 --num_images 2 --save_image --target_id -1 --data_path /home/lbw/Data
