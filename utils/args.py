import argparse

def parser_args():

    parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')

    parser.add_argument('--index', type=int, default="25",
                        help='the index for leaking images on CIFAR.')
    # parser.add_argument('--image', type=str,default="",
    #                     help='the path to customized image.')

    # ========================= deep leakage parameters ========================
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users: K")

    parser.add_argument('--epochs', type=int, default=300,
                        help='attack epochs')

    parser.add_argument('--epoch_interval', type=int, default=10,
                        help="interval for saving")

    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size for attack")

    parser.add_argument('--lr', type=float, default= 1,
                        help="learning rate for dlg")

    # ============================ Model arguments ===================================
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'resnet', 'lenet'],
                        help='model architecture name')
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help="name of dataset")

    parser.add_argument('--num_classes', default=10, type=int)
    
    parser.add_argument('--data_root', default="/home/lbw/Data",
                        help='dataset directory')

    # =========================== Other parameters ===================================
    parser.add_argument('--gpu', default='1', type=str)

    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')

    parser.add_argument('--iid', action='store_true', default =True,
                        help='dataset iid or not')

    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam]')

    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                         default='uniform', type=str,
                         help='which kind of sampling we use') 

    # =========================== DP ===================================
    parser.add_argument('--dp', action='store_true', default=False,
                        help='whether dp')

    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')

    
    args = parser.parse_args()

    return args
