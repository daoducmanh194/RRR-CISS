import os
import time
import random
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.utils import data

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from datasets.tasks import get_tasks
from datasets import VOCSegmentation, ADESegmentation
from metrics import StreamSegMetrics
import networks
import utils
from utils import ext_transforms as et
from utils.scheduler import PolyLR, WarmupPolyLR
from utils.loss import FocalLoss, BCEWithLogitsLossWithIgnoreIndex
from utils.utils import AverageMeter, RAdam, make_directories, print_time
from utils.memory import memory_sampling_balanced

torch.backends.cudnn.benchmark = True


def get_argparser():
    parser = argparse.ArgumentParser()

    # Config Options
    parser.add_argument('--config', type=str, default='./configs/config_cub_rrr.yaml', help='Config path')

    # Device Options
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help='device type')
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")

    # Experiments/Datasets Options
    parser.add_argument("--seed", type=int, default=1, help='Seed')
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012', help='path to Dataset')
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade'], help='Name of dataset')
    parser.add_argument("--task", type=str, default='15-1', help='cil task')
    parser.add_argument("--overlap", action='store_true', help='overlap setup (True), disjoint setup (False)')
    parser.add_argument("--num_classes", type=int, default=None, help='num classes (default: None)')


    # Deeplab model Options
    parser.add_argument("--backbone", type=str, default='resnet_101', 
                        choices=['resnet_50', 'resnet_101', 'mobilenet'], help='backbone name')
    parser.add_argument("--model", type=str, default='deeplabv3_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3_resnet50_separable',  'deeplabv3plus_resnet50_separable',
                                 'deeplabv3_resnet101_separable', 'deeplabv3plus_resnet101_separable',
                                 'deeplabv3_mobilenet_separable', 'deeplabv3plus_mobilenet_separable',
                                 'deeplabv3_resnet50_attention',  'deeplabv3plus_resnet50_attention',
                                 'deeplabv3_resnet101_attention', 'deeplabv3plus_resnet101_attention',
                                 'deeplabv3_mobilenet_attention', 'deeplabv3plus_mobilenet_attention'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--train_epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly',
                        choices=['poly', 'step', 'warm_poly'], help="learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False, help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str, help="restore from checkpoint")

    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10, help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100, help="epoch interval for eval (default: 100)")
    parser.add_argument("--train_optimizer", type=str, default='adam',
                        choices=['adam', 'radam', 'sgd'], help='train optimizer')

    # CIL options
    parser.add_argument("--pseudo", action='store_true', help="enable pseudo-labeling")
    parser.add_argument("--pseudo_thresh", type=float, default=0.7, help="confidence threshold for pseudo-labeling")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--mem_size", type=int, default=0, help="size of examplar memory")
    parser.add_argument("--freeze", action='store_true', help="enable network freezing")
    parser.add_argument("--bn_freeze", action='store_true', help="enable batchnorm freezing")
    parser.add_argument("--w_transfer", action='store_true', help="enable weight transfer")
    parser.add_argument("--unknown", action='store_true', help="enable unknown modeling")

    # Remember for Right Reasons Options
    parser.add_argument("--xai_memory", action='store_true', default=True, help="Use approaches RRR")
    parser.add_argument("--rrr_method", type=str, default='gc',
                        choices= ['gc', 'smooth','bp', 'gbp', 'deconv'], help="Select RRR methods")
    parser.add_argument("--rrr_upsample", action='store_true', default=False, help="Upsample for RRR")
    parser.add_argument("--rrr_loss", type=str, default='l1', choices=['l1', 'l2'], help="Loss use for RRR")
    parser.add_argument("--rrr_regularizer", type=int, default=100, help="Regulizer for RRR")
    parser.add_argument("--rrr_lr", type=float, default=0.0005, help="Learning rate for RRR")

    # Logger Options
    parser.add_argument("--wandb_log", action='store_true', default=True, help='Using logger for logging')
    parser.add_argument("--wandb_notes", type=str, default='', help='Wandb notes')
    parser.add_argument("--path_checkpoint", type=str, default='./logger_checkpoints', help='Path to checkpoint file')

    return parser


def get_dataset(args):
    """ Dataset And Augmentation
    """
    
    train_transform = et.ExtCompose([
        #et.ExtResize(size=args.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(args.crop_size, args.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if args.crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(args.crop_size),
            et.ExtCenterCrop(args.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        
    if args.dataset == 'voc':
        dataset = VOCSegmentation
    elif args.dataset == 'ade':
        dataset = ADESegmentation
    else:
        raise NotImplementedError
        
    dataset_dict = {}
    dataset_dict['train'] = dataset(args=args, image_set='train', transform=train_transform, cil_step=args.task_id)
    
    dataset_dict['val'] = dataset(args=args,image_set='val', transform=val_transform, cil_step=args.task_id)
    
    dataset_dict['test'] = dataset(args=args, image_set='test', transform=val_transform, cil_step=args.task_id)
    
    if args.task_id > 0 and args.mem_size > 0:
        dataset_dict['memory'] = dataset(args=args, image_set='memory', transform=train_transform, 
                                                 cil_step=args.task_id, mem_size=args.mem_size)

    return dataset_dict


def get_target_layer(args):
    if args.backbone == 'resnet18':
        args.target_layer = "features.7.1.conv2"
    elif args.backbone == 'resnet_50':
        args.target_layer = "features.7.1.conv2"
    elif args.backbone == 'resnet_101':
        args.target_layer = "module.classifier.head.0"
    elif args.backbone == 'mobilenet':
        args.target_layer = ""
    else:
        raise NotImplementedError
    return args.target_layer


def validate(args, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    # TODO: update return loss if necessary
    with torch.no_grad():
        for i, (images, labels, saliency_maps, file_names) in enumerate(loader):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            saliency_maps = saliency_maps.to(device, dtype=torch.float32, non_blocking=True)

            outputs = model(images)

            if args.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)

            # remove the unkown label
            if args.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)

        score = metrics.get_results()
    return score


def get_optimizer(args, task_id, lr=None):
    scheduler_opt  = None
    if (args.train_optimizer=="radam"):
        optimizer = RAdam(args.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    elif(args.train_optimizer=="adam"):
        optimizer= torch.optim.Adam(args.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0.0, amsgrad=False)
    elif(args.train_optimizer=="sgd"):
        optimizer= torch.optim.SGD(args.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
        scheduler_opt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                patience=5,
                                                                factor=3 / 10,
                                                                verbose=True)
    return optimizer, scheduler_opt


def get_optimizer_explanations(args, task_id, lr=None):
    scheduler_exp_opt = None
    if args.train_optimizer=="sgd":
        optimizer_explanations = torch.optim.SGD(args.model.parameters(), lr=lr, weight_decay=0.1)
        scheduler_exp_opt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_explanations,
                                                                        patience=5,
                                                                        factor=3/10,
                                                                        verbose=True)
    elif(args.train_optimizer == "adam"):
        optimizer_explanations = torch.optim.Adam(args.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                                weight_decay=0.0, amsgrad=False)

    elif (args.train_optimizer == "radam"):
        optimizer_explanations = RAdam(args.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
    return optimizer_explanations, scheduler_exp_opt


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    bn_freeze = args.bn_freeze if args.task_id > 0 else False
        
    target_cls = get_tasks(args.dataset, args.task, args.task_id)
    args.num_classes = [len(get_tasks(args.dataset, args.task, step)) for step in range(args.task_id+1)]
    if args.unknown: # re-labeling: [unknown, background, ...]
        args.num_classes = [1, 1, args.num_classes[0]-1] + args.num_classes[1:]
    fg_idx = 1 if args.unknown else 0
    
    curr_idx = [
        sum(len(get_tasks(args.dataset, args.task, step)) for step in range(args.task_id)), 
        sum(len(get_tasks(args.dataset, args.task, step)) for step in range(args.task_id+1))
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("==============================================")
    print(f"  task : {args.task}")
    print(f"  step : {args.task_id}")
    print("  Device: %s" % device)
    print( "  args : ")
    print(args)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    args.target_layer = get_target_layer(args)
    target_layer = "module.classifier.head."
    target_layer = target_layer + str(args.task_id+2)
    args.target_layer = target_layer
    # args.target_layer = None

    # TODO: Set up Approaches
    sal_loss = None
    if args.rrr_loss == "l1":
        sal_loss = torch.nn.L1Loss().to(device=device)
    elif args.rrr_loss == "l2":
        sal_loss = torch.nn.MSELoss().to(device=device)
    else:
        raise NotImplementedError
    
    # XAI
    if args.rrr_method == 'gc':
        print ("Using GradCAM to obtain saliency maps")
        from approaches.explanations import GradCAM as Explain
    elif args.rrr_method == 'smooth':
        print ("Using SmoothGrad to obtain saliency maps")
        from approaches.explanations import SmoothGrad as Explain
    elif args.rrr_method == 'bp':
        print ("Using BackPropagation to obtain saliency maps")
        from approaches.explanations import BackPropagation as Explain
    elif args.rrr_method == 'gbp':
        print ("Using Guided BackPropagation to obtain saliency maps")
        from approaches.explanations import GuidedBackPropagation as Explain
    elif args.rrr_method == 'deconv':
        from approaches.explanations import Deconvnet as Explain

    explainer = Explain(args)

    # optimizer_explanations, scheduler_exp_opt = get_optimizer_explanations(args, args.task_id)
    # optimizer, scheduler_opt = get_optimizer(args, args.task_id)
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': networks.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': networks.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': networks.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': networks.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': networks.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': networks.deeplabv3plus_mobilenet,
        'deeplabv3_resnet50_separable': networks.deeplabv3_resnet50_separable,
        'deeplabv3plus_resnet50_separable': networks.deeplabv3plus_resnet50_separable,
        'deeplabv3_resnet101_separable': networks.deeplabv3_resnet101_separable,
        'deeplabv3plus_resnet101_separable': networks.deeplabv3plus_resnet101_separable,
        'deeplabv3_mobilenet_separable': networks.deeplabv3_mobilenet_separable,
        'deeplabv3plus_mobilenet_separable': networks.deeplabv3plus_mobilenet_separable, 
        'deeplabv3_resnet50_attention': networks.deeplabv3_resnet50_attention,
        'deeplabv3plus_resnet50_attention': networks.deeplabv3plus_resnet50_attention,
        'deeplabv3_resnet101_attention': networks.deeplabv3_resnet101_attention,
        'deeplabv3plus_resnet101_attention': networks.deeplabv3plus_resnet101_attention,
        'deeplabv3_mobilenet_attention': networks.deeplabv3_mobilenet_attention,
        'deeplabv3plus_mobilenet_attention': networks.deeplabv3plus_mobilenet_attention,
    }

    model = model_map[args.model](num_classes=args.num_classes, output_stride=args.output_stride, bn_freeze=bn_freeze)
    if args.separable_conv and 'plus' in args.model:
        networks.convert_to_separable_conv(model.classifier)
    utils.utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    if args.task_id > 0:
        """ load previous model """
        model_prev = model_map[args.model](num_classes=args.num_classes[:-1], output_stride=args.output_stride, bn_freeze=bn_freeze)
        if args.separable_conv and 'plus' in args.model:
            networks.convert_to_separable_conv(model_prev.classifier)
        utils.utils.set_bn_momentum(model_prev.backbone, momentum=0.01)
    else:
        model_prev = None
    
    # Set up metrics
    metrics = StreamSegMetrics(sum(args.num_classes)-1 if args.unknown else sum(args.num_classes), dataset=args.dataset)

    print(model.classifier.head)
    
    # Set up optimizer & parameters
    if args.freeze and args.task_id > 0:
        for param in model_prev.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.head[-1].parameters(): # classifier for new class
            param.requires_grad = True
            
        training_params = [{'params': model.classifier.head[-1].parameters(), 'lr': args.lr}]
            
        if args.unknown:
            for param in model.classifier.head[0].parameters(): # unknown
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[0].parameters(), 'lr': args.lr})
            
            for param in model.classifier.head[1].parameters(): # background
                param.requires_grad = True
            training_params.append({'params': model.classifier.head[1].parameters(), 'lr': args.lr*1e-4})
        
    else:
        training_params = [{'params': model.backbone.parameters(), 'lr': 0.001},
                           {'params': model.classifier.parameters(), 'lr': 0.01}]
        
    optimizer = torch.optim.SGD(params=training_params, 
                                lr=args.lr, 
                                momentum=0.9, 
                                weight_decay=args.weight_decay,
                                nesterov=True)
    optimizer_explanations = torch.optim.Adam(params=training_params, lr=args.rrr_lr, betas=(0.9, 0.999), eps=1e-08,
                                                weight_decay=0.0, amsgrad=False)
    
    print("----------- trainable parameters --------------")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    print("-----------------------------------------------")
    
    def save_ckpt(path):
        torch.save({
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.utils.mkdir('checkpoints')
    # Restore
    best_score = -1
    cur_itrs = 0
    cur_epochs = 0
    
    if args.overlap:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_overlap.pth"
    else:
        ckpt_str = "checkpoints/%s_%s_%s_step_%d_disjoint.pth"
    
    if args.task_id > 0: # previous step checkpoint
        args.ckpt = ckpt_str % (args.model, args.dataset, args.task, args.task_id-1)
        
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))["model_state"]
        model_prev.load_state_dict(checkpoint, strict=True)
        
        if args.unknown and args.w_transfer:
            # weight transfer : from unknonw to new-class
            print("... weight transfer")
            curr_head_num = len(model.classifier.head) - 1

            checkpoint[f"classifier.head.{curr_head_num}.0.weight"] = checkpoint["classifier.head.0.0.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.weight"] = checkpoint["classifier.head.0.1.weight"]
            checkpoint[f"classifier.head.{curr_head_num}.1.bias"] = checkpoint["classifier.head.0.1.bias"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_mean"] = checkpoint["classifier.head.0.1.running_mean"]
            checkpoint[f"classifier.head.{curr_head_num}.1.running_var"] = checkpoint["classifier.head.0.1.running_var"]

            last_conv_weight = model.state_dict()[f"classifier.head.{curr_head_num}.3.weight"]
            last_conv_bias = model.state_dict()[f"classifier.head.{curr_head_num}.3.bias"]

            for i in range(args.num_classes[-1]):
                last_conv_weight[i] = checkpoint["classifier.head.0.3.weight"]
                last_conv_bias[i] = checkpoint["classifier.head.0.3.bias"]

            checkpoint[f"classifier.head.{curr_head_num}.3.weight"] = last_conv_weight
            checkpoint[f"classifier.head.{curr_head_num}.3.bias"] = last_conv_bias

        model.load_state_dict(checkpoint, strict=False)
        print("Model restored from %s" % args.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")

    model = nn.DataParallel(model)
    mode = model.to(device)
    mode.train()
    
    if args.task_id > 0:
        model_prev = nn.DataParallel(model_prev)
        model_prev = model_prev.to(device)
        model_prev.eval()

        if args.mem_size > 0:
            memory_sampling_balanced(args, model_prev)
            
        # Setup dataloader
    if not args.crop_val:
        args.val_batch_size = 1
    
    dataset_dict = get_dataset(args)
    train_loader = data.DataLoader(
        dataset_dict['train'], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        dataset_dict['val'], batch_size=args.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(
        dataset_dict['test'], batch_size=args.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print("Dataset: %s, Train set: %d, Val set: %d, Test set: %d" %
          (args.dataset, len(dataset_dict['train']), len(dataset_dict['val']), len(dataset_dict['test'])))
    
    if args.task_id > 0 and args.mem_size > 0:
        memory_loader = data.DataLoader(
            dataset_dict['memory'], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    total_itrs = args.train_epoch * len(train_loader)
    val_interval = max(100, total_itrs // 100)
    print(f"... train epoch : {args.train_epoch} , iterations : {total_itrs} , val_interval : {val_interval}")
        
    #==========   Train Loop   ==========#
    if args.test_only:
        model.eval()
        test_score = validate(args=args, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        
        print(metrics.to_str(test_score))
        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())

        first_cls = len(get_tasks(args.dataset, args.task, 0)) # 15-1 task -> first_cls=16
        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))
        return

    if args.lr_policy=='poly':
        scheduler = PolyLR(optimizer, total_itrs, power=0.9)
    elif args.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.lr_policy=='warm_poly':
        warmup_iters = int(total_itrs*0.1)
        scheduler = WarmupPolyLR(optimizer, total_itrs, warmup_iters=warmup_iters, power=0.9)

    # Set up criterion
    if args.loss_type == 'focal_loss':
        criterion = FocalLoss(ignore_index=255, size_average=True)
    elif args.loss_type == 'ce_loss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif args.loss_type == 'bce_loss':
        criterion = BCEWithLogitsLossWithIgnoreIndex(ignore_index=255, 
                                                           reduction='mean')
        
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    avg_loss = AverageMeter(object)
    avg_time = AverageMeter(object)
    
    model.train()
    save_ckpt(ckpt_str % (args.model, args.dataset, args.task, args.task_id))
    
    # =====  Train  =====
    while cur_itrs < total_itrs:
        cur_itrs += 1
        optimizer.zero_grad()
        end_time = time.time()
        
        """ data load """
        try:
            images, labels, sal_maps, _ = next(train_iter)
        except:
            train_iter = iter(train_loader)
            images, labels, sal_maps, _ = next(train_iter)
            cur_epochs += 1
            avg_loss.reset()
            avg_time.reset()
            
            with torch.cuda.amp.autocast(enabled=args.amp):

                images = images.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                sal_maps = sal_maps.to(device, dtype=torch.long, non_blocking=True)

                if args.task_id > 0 and args.xai_memory:
                    explanations, model, _, _ = explainer(images, model, args.task_id)
                    # saliency_size = explainer.size()

                    # print(explanations.shape)
                    # print(images.shape)
                    explanations = explanations.view(explanations.size(0), -1)
                    sal_images = images.view(images.size(0), -1)
                    # ex = torch.cat((explanations, explanations), 1)
                    # # explanations = explanations.expand(images.shape[0], images.shape[1])
                    # for i in range(images.shape[1] - explanations.shape[1]):
                    #     ex = torch.cat((ex, explanations), 1)
                    explanations = explanations.to(device)
                    new_ex = torch.ones(sal_images.shape, device=explanations.device)
                    for i in range(sal_images.shape[1] // explanations.shape[1]):
                        new_ex.index_copy_(1, torch.tensor(range(i * explanations.shape[1], (i + 1) * explanations.shape[1]), device=device), explanations)
                    # print(new_ex)
                    # print(images)
                    saliency_loss = sal_loss(new_ex, sal_images)
                    saliency_loss *= 0.0005
                    # print("Saliency loss: {}".format(saliency_loss))

                    # try:
                    #     saliency_loss.requires_grad = True
                    # except:
                    #     continue

                    optimizer_explanations.zero_grad()
                    saliency_loss.backward(retain_graph=True)
                    optimizer_explanations.step()

            # scaler.scale(saliency_loss).backward()
            # scaler.step(optimizer_explanations)
            # scaler.update()

            # scheduler.step()
            
            
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        sal_maps = sal_maps.to(device, dtype=torch.long, non_blocking=True)
            
        """ memory """
        if args.task_id > 0 and args.mem_size > 0:
            try:
                m_images, m_labels, m_sal_maps, _ = next(mem_iter)
            except:
                mem_iter = iter(memory_loader)
                m_images, m_labels, m_sal_maps, _ = next(mem_iter)

            m_images = m_images.to(device, dtype=torch.float32, non_blocking=True)
            m_labels = m_labels.to(device, dtype=torch.long, non_blocking=True)
            m_sal_maps = m_sal_maps.to(device, dtype=torch.long, non_blocking=True)
            
            rand_index = torch.randperm(args.batch_size)[:args.batch_size // 2].cuda()
            images[rand_index, ...] = m_images[rand_index, ...]
            labels[rand_index, ...] = m_labels[rand_index, ...]
            sal_maps[rand_index, ...] = m_sal_maps[rand_index, ...]

        
        """ forwarding and optimization """
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(images)

            if args.pseudo and args.task_id > 0:
                """ pseudo labeling """
                with torch.no_grad():
                    outputs_prev = model_prev(images)

                if args.loss_type == 'bce_loss':
                    pred_prob = torch.sigmoid(outputs_prev).detach()
                else:
                    pred_prob = torch.softmax(outputs_prev, 1).detach()
                    
                pred_scores, pred_labels = torch.max(pred_prob, dim=1)
                pseudo_labels = torch.where( (labels <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= args.pseudo_thresh), 
                                            pred_labels, 
                                            labels)
                    
                loss = criterion(outputs, pseudo_labels)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        avg_loss.update(loss.item())
        avg_time.update(time.time() - end_time)
        end_time = time.time()

        if (cur_itrs) % 10 == 0:
            print("[%s / step %d] Epoch %d, Itrs %d/%d, Loss=%6f, Time=%.2f , LR=%.8f" %
                  (args.task, args.task_id, cur_epochs, cur_itrs, total_itrs, 
                   avg_loss.avg, avg_time.avg*1000, optimizer.param_groups[0]['lr']))

        if val_interval > 0 and (cur_itrs) % val_interval == 0:
            print("validation...")
            model.eval()
            val_score = validate(args=args, model=model, loader=val_loader, 
                                 device=device, metrics=metrics)
            print(metrics.to_str(val_score))
            
            model.train()
            
            class_iou = list(val_score['Class IoU'].values())
            val_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] + [class_iou[0]])
            curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
            print("curr_val_score : %.4f" % (curr_score))
            print()
            
            if curr_score > best_score:  # save best model
                print("... save best ckpt : ", curr_score)
                best_score = curr_score
                save_ckpt(ckpt_str % (args.model, args.dataset, args.task, args.task_id))


    print("... Training Done")
    
    if args.task_id > 0:
        print("... Testing Best Model")
        best_ckpt = ckpt_str % (args.model, args.dataset, args.task, args.task_id)
        
        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        
        test_score = validate(args=args, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        print(metrics.to_str(test_score))

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())
        first_cls = len(get_tasks(args.dataset, args.task, 0))

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


if __name__ == '__main__':
            
    args = get_argparser().parse_args()

    tstart = time.time()
    utils.utils.print_time(start=True)
    args.path_checkpoint, args.wandb_notes = utils.utils.make_directories(args)

    # if args.wandb_log:
    #     wandb.init(project='RRR',name=args.wandb_notes,
    #                config=args.config, notes=args.wandb_notes,
    #                allow_val_change=True)

    utils.utils.save_code(args)
    print('=' * 100)
    print('Arguments = ')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)
        
    start_step = 0
    total_step = len(get_tasks(args.dataset, args.task))
    
    for step in range(start_step, total_step):
        args.task_id = step
        main(args)

    print("All Done!")
    print('[Elapsed time = {:.1f} min - {:0.1f} hours]'.format((time.time() - tstart)/(60), (time.time() - tstart)/(3600)))
    utils.utils.print_time(start=False)
        