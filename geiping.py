"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py
"""
import torch
import torchvision
import numpy as np
from PIL import Image
import inversefed
from inversefed.reconstruction_algorithms import GradientReconstructor
from inversefed.test_grad import GradientReconstructor_test
import datetime
import time
import os
from utils.datasets import get_data
from torch.utils.data import DataLoader
from models.vision import LeNet, LeNet_Imagenet, AlexNet_Imagenet, AlexNet_Cifar, ResNet18
from utils.metrics import Loss, label_to_onehot, cross_entropy_for_onehot, Classification, psnr

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Parse input arguments
args = inversefed.options().parse_args()

# Parse training strategy
defs = inversefed.training_strategy("conservative")
defs.epochs = args.epochs

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
imagenet_mean = [0.4802, 0.4481, 0.3975] 
imagenet_std = [0.2302, 0.2265, 0.2262]

if __name__ == "__main__":
    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Prepare for training
    # Get data:
    #loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)
    dst, test_set = get_data(dataset=args.dataset,
                                                    data_root=args.data_path,
                                                    normalized=True)
    local_train_ldr = DataLoader(dst, batch_size = 32, shuffle=False, num_workers=2)
    if args.dataset == "imagenet":
        num_classes = 20
        input_size = 224
    if args.dataset == "cifar10":
        num_classes = 10
        input_size = 32
    if args.dataset == "cifar100":
        num_classes = 100
        input_size = 32

    # mean, std
    dm = torch.as_tensor(imagenet_mean, **setup)[:, None, None]
    ds = torch.as_tensor(imagenet_std, **setup)[:, None, None]

    # Prepare Model
    if args.model == "lenet":
        if args.dataset == "imagenet":
            net = LeNet_Imagenet(input_size=input_size).to(**setup)
        else:
            net = LeNet(input_size=input_size).to(**setup)
    if args.model == "alexnet":
        if args.dataset == "imagenet":
            net = AlexNet_Imagenet(num_classes=num_classes, input_size = input_size).to(**setup) # pretrained = True
        else: 
            net = AlexNet_Cifar(num_classes=num_classes, input_size = input_size).to(**setup)
    if args.model == "resnet":
        if args.dataset == "imagenet":
            net = torchvision.models.resnet18(True)
            #print(net_)
            #net = ResNet18(num_classes=num_classes, imagenet = True).to(**setup)
            #print(net)
            #net = torchvision.models.resnet18(num_classes =20, pretrained=False).to(**setup)
        else:
            net = ResNet18(num_classes=num_classes, imagenet = False).to(**setup)
    model = net 
    model.to(**setup)
    
    model.eval()

    # Sanity check: Validate model accuracy
    # Choose example images from the validation set or from third-party sources
    target_id = args.target_id +  args.num_images

 
    if args.num_images == 1:
        ground_truth, labels = local_train_ldr.dataset[target_id]
      
        ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])
        ground_truth = torch.as_tensor(
                np.array(Image.open("auto.jpg").resize((224, 224), Image.BICUBIC)) / 255, device ="cuda", dtype=torch.float
            )
        ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
        target_id_ = target_id + 1
    else:
        ground_truth, labels = [], []
        target_id_ = target_id
        while len(labels) < args.num_images:
            img, label = local_train_ldr.dataset[target_id_]
            target_id_ += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)

      
    img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

    # if args.num_images == 1: 
    #     # demo image
    #     # Specify PIL filter for lower pillow versions
    #     ground_truth = torch.as_tensor(np.array(Image.open("auto.jpg").resize((224, 224), Image.BICUBIC)) / 255, **setup)
    #     ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
    #     labels = torch.as_tensor((1,), device=setup["device"])
    #     target_id = -1
    #     img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

    # else:
    #     ground_truth, labels = [], []
    #     if args.target_id is None:
    #         target_id = np.random.randint(len(validloader.dataset))
    #     else:
    #         target_id = args.target_id
    #     while len(labels) < args.num_images:
    #         img, label = validloader.dataset[target_id]
    #         target_id += 1
    #         if label not in labels:
    #             labels.append(torch.as_tensor((label,), device=setup["device"]))
    #             ground_truth.append(img.to(**setup))
    #     ground_truth = torch.stack(ground_truth)
    #     labels = torch.cat(labels)
    #     if args.label_flip:
    #         labels = (labels + 1) % len(trainloader.dataset.classes)
    #     img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

    # Run reconstruction  accumulation 0 for gradient
    if args.accumulation == 0:
        model.zero_grad()
        loss_fn = Classification() 

        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        print(f"Full gradient norm is {full_norm:e}.")

        if args.optim == "ours":
            config = dict(
                signed=args.signed,
                boxed=args.boxed,
                cost_fn=args.cost_fn,
                indices="def",
                weights="equal",
                lr=0.1,
                optim=args.optimizer,
                restarts=args.restarts,
                max_iterations=24000,
                total_variation=args.tv,
                init="randn",
                filter="none",
                lr_decay=True,
                scoring_choice="loss",
            )
        elif args.optim == "zhu":
            config = dict(
                signed=False,
                boxed=False,
                cost_fn="l2",
                indices="def",
                weights="equal",
                lr=1e-4,
                optim="LBFGS",
                restarts=args.restarts,
                max_iterations=300,
                total_variation=args.tv,
                init=args.init,
                filter="none",
                lr_decay=False,
                scoring_choice=args.scoring_choice,
            )

        config = dict( 
            signed = True, 
            boxed = True, 
            cost_fn = "sim", 
            lr = 0.1, 
            indices = 'def', 
            weights = 'equal', 
            optim = 'adam', 
            normalized = True, 
            restarts = 1, 
            epochs = 10000, 
            interval = 500, 
            total_variation = 0.0001, 
            bn_stat = 0, 
            image_norm = 0, 
            group_lazy = 0, 
            avg_type = 'median', 
            init = 'randn', 
            filter = 'none', 
            scoring_choice = 'loss', 
            lr_decay = True)
        
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)
      
        # rec_machine = GradientReconstructor_test(model, (dm, ds), config, num_images=args.num_images)
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)

    # else:
    #     #for fedavg
    #     output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, dryrun=args.dryrun)

    # Compute stats
    test_mse = (output - ground_truth).pow(2).mean().item()
    feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    test_psnr = psnr(output, ground_truth, factor=1 / ds)

    # Save the resulting image
    if args.save_image and not args.dryrun:
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)

        for j in range(args.num_images):
            # reconstructed images
            filename = f'{"trained" if args.trained_model else ""}'f"{args.model}_{args.cost_fn}_{j}.png"

            torchvision.utils.save_image(output_denormalized[j:j + 1, ...],
                                                    os.path.join(f'images/', filename))

            gt_filename = f"ground_truth-{j}.png"
            torchvision.utils.save_image(gt_denormalized[j:j + 1, ...], os.path.join(f'images/', gt_filename))

        # rec_filename = (
        #     f'{local_train_ldr.dataset.classes[labels][0]}_{"trained" if args.trained_model else ""}'
        #     f"{args.model}_{args.cost_fn}-{args.target_id}.png"
        # )
        # torchvision.utils.save_image(output_denormalized, os.path.join(args.image_path, rec_filename))

    # Save to a table:
    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

    inversefed.utils.save_to_table(
        args.table_path,
        name=f"exp_{args.name}",
        dryrun=args.dryrun,
        model=args.model,
        dataset=args.dataset,
        trained=args.trained_model,
        accumulation=args.accumulation,
        restarts=args.restarts,
        OPTIM=args.optim,
        cost_fn=args.cost_fn,
        indices=args.indices,
        weights=args.weights,
        scoring=args.scoring_choice,
        init=args.init,
        tv=args.tv,
        rec_loss=stats["opt"],
        psnr=test_psnr,
        test_mse=test_mse,
        feat_mse=feat_mse,
        target_id=target_id,
        seed=1234,
        timing=str(datetime.timedelta(seconds=time.time() - start_time)),
        dtype=setup["dtype"],
        epochs=defs.epochs,
        val_acc=None,
        #rec_img=rec_filename,
        gt_img=gt_filename,
    )

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print("---------------------------------------------------")
    print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
    print("-------------Job finished.-------------------------")
