import argparse

def parser_args():

    parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')

    parser.add_argument('--index', type=int, default="0",
                        help='the index for leaking images on CIFAR.')
    # parser.add_argument('--image', type=str,default="",
    #                     help='the path to customized image.')

    # ========================= deep leakage parameters ========================

    parser.add_argument('--epochs', type=int, default=300,
                        help='attack epochs')

    parser.add_argument('--epoch_interval', type=int, default=10,
                        help="interval for saving")

    parser.add_argument('--bs', type=int, default=4,
                        help="batch size for attack")

    parser.add_argument('--lr', type=float, default= 1,
                        help="learning rate for dlg")

    parser.add_argument('--cost_fn', default='sim',  choices=['simlocal', 'l2', 'sim'], type=str, help='Choice of cost function.')

    parser.add_argument('--tv', type=float, default= 1e-4,
                        help="TV loss parameter for dlg")

    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam, LBFGS, geiping, gaussian, GC, BN, Zhu]')

    # ============================ Model arguments ===================================
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'resnet', 'lenet'],
                        help='model architecture name')
    
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'trained'],
                        help='trained model or random model')
    
    parser.add_argument('--attack_iters', type=int, default=10,
                        help="iteration number for attack")
    parser.add_argument('--start_attack_iters', type=int, default=0,
                        help="start iteration number for attack")
    parser.add_argument('--avg_type', default='avg', type=str)

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist', 'imagenet'], help="name of dataset")

    parser.add_argument('--num_classes', default=10, type=int)
    
    parser.add_argument('--data_root', default="/home/lbw/Data",
                        help='dataset directory')

    # =========================== Other parameters ===================================
    parser.add_argument('--gpu', default='1', type=str)

    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')

    parser.add_argument('--normalized', action='store_true', default =False,
                        help='normalized or not')

    
    args = parser.parse_args()

    return args
