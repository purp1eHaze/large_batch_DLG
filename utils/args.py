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
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate for inner update")
    parser.add_argument('--iid', action='store_true', default =True,
                        help='dataset iid or not')
    parser.add_argument('--wd', type=float, default=4e-5,
                        help='weight decay')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='communication round')
    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                         default='uniform', type=str,
                         help='which kind of client sampling we use') 
    
    # ============================ Model arguments ===================================
    parser.add_argument('--model_name', type=str, default='alexnet', choices=['alexnet', 'resnet'],
                        help='model architecture name')
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help="name of dataset")

    parser.add_argument('--num_classes', default=10, type=int)
    
    parser.add_argument('--data_root', default='/data/home/brannli/data/',
                        help='dataset directory')

    # =========================== Other parameters ===================================
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--log_interval', default=1, type=int,
                        help='interval for evaluating loss and accuracy')
    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')

    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')



    # =========================== DP ===================================
    parser.add_argument('--dp', action='store_true', default=False,
                        help='whether dp')

    parser.add_argument('--sigma',  type=float, default= 0.1 , help='the sgd of Gaussian noise')

    
    args = parser.parse_args()

    return args
