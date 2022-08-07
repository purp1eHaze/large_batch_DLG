from locale import normalize

def Setup_Config(args):

    if args.optim == "adam": 
        config = Adam_Config(args) 
    if args.optim == "LBFGS":     
        config = LBFGS_Config(args)
    if args.optim == "SGD":
        config = SGD_Config(args)
    if args.optim == "geiping":
        config = Geiping_Config(args)
    if args.optim == "BN":
        config = BN_Config(args)
    if args.optim == "GC":
        config = GC_Config(args)
    if args.optim == "gaussian":
        config = Gaussian_Config(args)
    if args.optim == "Zhu":
        config = Zhu_Config(args)
    return config

def LBFGS_Config(args): 
    return dict(
            boxed=False,
            cost_fn="l2",
            lr=1,
            optim="LBFGS",
            normalized = False,
            epochs=100,
            interval= 10,
            TV_para = args.tv,
            avg_type = "median",
            lr_decay = 0.3)

def Adam_Config(args): 
    return dict(
            boxed=False,
            cost_fn="l2",
            lr=0.1,
            optim="adam",
            normalized = False,
            epochs= 6000,
            interval= 500,
            TV_para = args.tv,
            avg_type = "median",
            lr_decay = 0.1)

def SGD_Config(args): 
    return dict(
            boxed=False,
            cost_fn="l2",
            lr=0.01,
            optim="sgd",
            normalized = False,
            epochs= 6000,
            interval= 500,
            TV_para = args.tv,
            avg_type = "median",
            lr_decay = 0.1)

def Geiping_Config(args): 
    return dict(
            signed=True,  # Gradient Sign args.signed
            boxed= True,   #args.boxed,
            cost_fn="sim",
            lr=0.1,
            indices="def",
            weights="equal",
            optim="adam",
            normalized = True,
            restarts=1, #args.restarts,
            epochs= 10000,
            interval= 500,
            #total_variation= 0,
            total_variation=1e-4, #args.tv,
            bn_stat=0,
            image_norm=0,
            group_lazy=0,

            avg_type = "median",
            init="randn",
            filter="none",
            scoring_choice="loss",
            lr_decay=True)

def Zhu_Config(args):
    return dict(signed=True,
                boxed=True,
                cost_fn="l2",
                indices="def",
                weights="equal",
                lr= 0.1,
                optim="LBFGS",
                normalized = False,
                restarts=1,
                epochs= 200,
                interval= 10,
                total_variation=0,
                bn_stat=0,
                image_norm=0,
                group_lazy=0,

                init="randn",
                filter="none",
                lr_decay=False,
                scoring_choice="loss",
    )

def Gaussian_Config(args):
    return dict(signed=True,
                boxed=True,
                cost_fn="gaussian",
                indices="def",
                weights="exp",
                lr= 0.001,
                optim="adam",
                normalized = True,
                restarts=1,
                epochs= 10000,
                interval= 100,
                total_variation=0,
                #total_variation=1e-4,
                bn_stat=0,
                image_norm=0,
                group_lazy=0,
                init="randn",
                filter="none",
                lr_decay=False,
                scoring_choice="loss",
    )

def BN_Config(args):
    return dict(signed=True,
                boxed=True,
                cost_fn='l2',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                normalized = True,
                restarts=1,
                epochs= 6000,
                interval= 500,
                total_variation=0.0001,
                bn_stat=0.1,
                image_norm=0,
                group_lazy=0,
                init='randn',
                lr_decay=True,
                dataset=args.dataset,
                )

def GC_Config(args):
    return dict(signed=True,
                boxed=True,
                cost_fn='l2',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                normalized = False,
                restarts=10,
                epochs= 6000,
                interval= 500,
                total_variation=0.0001,
                bn_stat=0,
                image_norm=0,
                group_lazy=0.01,
                init='randn',
                lr_decay=True,
                dataset=args.dataset,
                )
