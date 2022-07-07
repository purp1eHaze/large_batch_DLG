LBFGS_Config = dict(
    boxed=False,
    cost_fn="l2",
    lr=1,
    optim="LBFGS",
    # restarts=args.restarts,
    # epochs=300,
    # interval= 10,
    # TV_para = args.tv,
    avg_type = "median",
    lr_decay = 0.3
)

Adam_Config = dict(
    boxed=False,
    cost_fn="l2",
    lr=0.1,
    optim="adam",
    # restarts=args.restarts,
    # epochs= 6000,
    # interval= 500,
    # TV_para = args.tv,
    avg_type = "median",
    lr_decay = 0.1
)

SGD_Config = dict(
    boxed=False,
    cost_fn="l2",
    lr=0.01,
    optim="sgd",
    # restarts=args.restarts,
    # epochs= 6000,
    # interval= 500,
    # TV_para = args.tv,
    avg_type = "median",
    lr_decay = 0.1
)

Geiping_Config = dict(
    boxed=False,
    cost_fn="sim",
    lr=1,
    optim="adam",
    # restarts=args.restarts,
    # epochs=6000,
    # interval= 500,
    # TV_para = args.tv,
    avg_type = "median",
    lr_decay = 0.1
)