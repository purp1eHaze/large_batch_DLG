from locale import normalize


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
            total_variation=1e-4, #args.tv,
            avg_type = "median",
            init="randn",
            filter="none",
            scoring_choice="loss",
            lr_decay=True)
