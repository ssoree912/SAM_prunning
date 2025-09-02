# Standard library imports
import os
import sys
import time
import random
import argparse
import datetime
import json
from pathlib import Path

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

# Timm-related imports
import timm
# Mixup is not typically used with CIFAR in the same way as ImageNet,
# but included for structural similarity if args enable it.
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# Timm scheduler/optimizer create functions might not be used if using standard PyTorch ones
# from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

# Local application imports
# Assuming these are general utility functions and distributed setup
from utils import *
# Assuming engine.py contains train_one_epoch and evaluate compatible with script 1
from engine import *
# Assuming pruning.py contains MaskLinear and related functions
from pruning import *
# Assuming config.py provides necessary arguments
from config import get_train_args
from SAM import SAM

# === Helper Functions/Classes (from Script 1) ===

def replace_all_linear_layers(module):
    """
    Recursively replaces all nn.Linear layers in a module with MaskLinear layers.
    Copies weights and biases from the original linear layer.
    """
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Linear):
            # Create a new MaskLinear layer with the same dimensions
            new_module = MaskLinear(sub_module.in_features, sub_module.out_features)
            with torch.no_grad():
                # Copy weights
                new_module.weight.copy_(sub_module.weight)
                # Copy bias if it exists
                if sub_module.bias is not None:
                    new_module.bias.copy_(sub_module.bias)
            # Replace the original layer with the new MaskLinear layer
            setattr(module, name, new_module)
        else:
            # Recursively apply to child modules
            replace_all_linear_layers(sub_module)

def set_dropout_to_zero(module):
    """
    Recursively sets the dropout probability (p) of all nn.Dropout layers to 0.0.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            # Replace Dropout layer with one having p=0.0
            setattr(module, name, nn.Dropout(p=0.0))
        else:
            # Recursively apply to child modules
            set_dropout_to_zero(child)

class VitWrapper(nn.Module):
    """
    A wrapper class for the Vision Transformer model.
    Provides methods to interact with MaskLinear layers within the model.
    """
    def __init__(self, model):
        super(VitWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """ Standard forward pass. """
        # Note: The original PromptedVisionTransformer returned outputs,
        # this simplified wrapper assumes the base model returns the final output.
        # Adjust if intermediate outputs are needed.
        return self.model(x)

    def set_all_type_values(self, type_value):
        """ Sets the 'type_value' for all MaskLinear layers in the model. """
        for module in self.model.modules():
            if isinstance(module, MaskLinear):
                module.set_type_value(type_value)
                
def setattr_nested(obj, name, value):
    names = name.split('.')
    for i in range(len(names)-1):
        obj = getattr(obj, names[i])
    setattr(obj, names[-1], value)
                
def remove_module_prefix(state_dict):
    return {k.replace('model.', ''): v for k, v in state_dict.items()}


def hyperparam():
    args = config.config()
    return args

def main(args):
    # --- Distributed Training Setup ---
    # Set CUDA device visibility based on arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    # Initialize distributed mode (if enabled)
    init_distributed_mode(args)

    print("Effective Arguments:", args)
    device = torch.device(args.device)
    # Enable cuDNN benchmark mode for potentially faster training
    cudnn.benchmark = True
    # ====================================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 이 부분을 수정합니다 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # ====================================================================

    # 1. my_dataloader.py에서 커스텀 데이터셋을 import 합니다.
    # 파일 경로에 맞게 수정해야 할 수 있습니다.
    from my_dataloader import CIFAR100 

    # 2. 여러 개의 transform 파이프라인을 담을 리스트를 생성합니다.
    train_transforms = []
    
    # CIFAR-100 정규화 통계
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    # 3. args.aug_nums 만큼 반복하여 transform 리스트를 채웁니다.
    #    Multi-view를 위해 args.aug_nums를 2로 설정해야 합니다.
    for _ in range(2):
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_transforms.append(train_transform)

    # 테스트용 단일 transform
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # 4. 커스텀 CIFAR100 데이터셋을 호출하고, `transform_list` 인자를 사용합니다.
    dataset_train = CIFAR100(
        root=args.data_path, 
        train=True,
        transform_list=train_transforms, # transform 대신 transform_list 사용
        download=True
    )
    # valset은 torchvision의 기본 클래스를 그대로 사용해도 무방합니다.
    dataset_val = torchvision.datasets.CIFAR100(
        root=args.data_path, 
        train=False,
        download=True, 
        transform=transform_val
    )
    
    args.nb_classes = 100
    
    # ====================================================================
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 수정 끝 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ====================================================================
    # Load CIFAR-100 dataset

    # --- Samplers (Distributed or Standard) ---
    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        # Note: RASampler might need adaptation for CIFAR or use standard DistributedSampler
        # Using DistributedSampler as per script 1's structure
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        # Standard samplers if not using distributed training
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # --- DataLoaders ---
    print("dataloader_")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=False,
        drop_last=True, # Drop last incomplete batch
    )

    # Note: Script 1 had specific logic for ThreeAugment. If needed, adapt here.
    # if args.ThreeAugment:
    #     data_loader_train.dataset.transform = new_data_aug_generator(args) # Requires adaptation for CIFAR

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # Using args.batch_size for consistency, script 1 used 1.5*batch_size
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=False,
        drop_last=False
    )

    # --- Mixup Setup (from Script 1) ---
    mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     print("Mixup/Cutmix Enabled")
    #     mixup_fn = Mixup(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.nb_classes)
    # else:
    #     print("Mixup/Cutmix Disabled")


    # --- Model Definition and Loading ---
    # Model configurations (same as script 1, potentially adjust checkpoints if needed)
    model_configs = {
       'tiny': {
           'model': 'vit_tiny_patch16_224',
           'checkpoint': 'checkpoint/deit_tiny_patch16_224-a1311bcf.pth',
           # 'prompt_dim': 192 # Removed prompt specific info
       },
       'small': {
           'model': 'vit_small_patch16_224',
           'checkpoint': 'checkpoint/deit_small_patch16_224-cd65a155.pth',
           # 'prompt_dim': 384 # Removed prompt specific info
       },
       'base': {
           'model': 'vit_base_patch16_224',
           'checkpoint': 'checkpoint/deit_base_patch16_224-b5f2ef4d.pth',
           # 'prompt_dim': 768 # Removed prompt specific info
       }
    }

    model_name = model_configs[args.model_size]['model']
    # checkpoint_path = model_configs[args.model_size]['checkpoint']

    print(f"Creating model: {model_name}")
    # Create model with CIFAR-100 number of classes
    model = timm.create_model(model_name, pretrained=False, num_classes=100)


    # Use GPU if available
    replace_all_linear_layers(model) 
    set_dropout_to_zero(model) 

    state_dict = torch.load('./output/' + args.checkpoint , weights_only=False)

    state_dict = remove_module_prefix(state_dict['model'])
    model.load_state_dict(state_dict)

    # Wrap model
    model = VitWrapper(model)
    model.to(device)
    model.set_all_type_values(0)
    
    if args.distributed:
        # Use DistributedDataParallel for distributed training
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # Ensure type value is set after DDP wrapping if using module attribute access
        model.module.set_all_type_values(0) # Assuming initial type value is 0
        model_without_ddp = model.module
    else:
        # Use DataParallel for single-node multi-GPU (or single GPU)
        # Note: Script 1 used DDP even for single node if args.distributed was true.
        # Sticking to DDP if distributed flag is set.
        # If not distributed, no parallel wrapper needed unless multiple GPUs specified via CUDA_VISIBLE_DEVICES
        # and args.gpu is not set (or handled differently in init_distributed_mode)
        if torch.cuda.device_count() > 1 and not args.distributed:
             print(f"Using DataParallel for {torch.cuda.device_count()} GPUs.")
             # Ensure gpuids are correctly parsed if using DataParallel
             # gpu_ids = list(map(int, args.cu_num.split(',')))
             # model = nn.DataParallel(model, device_ids=gpu_ids)
             # model.module.set_all_type_values(0) # Access via module
             # model_without_ddp = model.module
             # For simplicity, assuming single GPU if not distributed, or DDP handles it.
             model.set_all_type_values(0) # Direct access if no wrapper
             model_without_ddp = model # No DDP wrapper
        else:
             model.set_all_type_values(0) # Direct access if no wrapper
             model_without_ddp = model # No DDP wrapper
    rho = args.rho
    print("SAM rho 지정 바로 아래줄")
    print(rho)
    print()

    optimizer = SAM(model.parameters(), 
                    AdamW, 
                    rho=rho,                     # SAM neighbourhood radius
                    adaptive=False,               # ASAM 여부
                    v2=True,                       # one-step SAM(v2) 여부
                    lr=args.lr,                           # AdamW 에 들어갈 lr
                    betas=(0.9, 0.999),                   # AdamW 전용 인자
                    eps=1e-8,
                    weight_decay=args.weight_decay,
                          )       # AdamW decoupled WD                       
                     
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    # global iterations
    iterations = 0
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)



        if args.use_DLB == 0: #원본 sam v2 실험 
            train_stats = fine_train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, None, None,
                set_training_mode=args.train_mode, # or False if not used
                args=args# Pass args for pruning logic inside train_one_epoch
            )
            
        elif args.use_DLB== 1: 
            train_stats = fine_train_one_epoch_DLB_test(
                model,criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, None, None,
                set_training_mode=args.train_mode, # or False if not used
                args=args # Pass args for pruning logic inside train_one_epoch
            )                                                                           


        lr_scheduler.step(epoch)
        
        print(f'Current learning rate: {lr_scheduler.get_last_lr()[0]}')
        # print(f'Sparsity:{target_sparsity}' )


        # if args.output_dir and (epoch+1) == args.target_epoch: 
        
        # if args.output_dir : 
        #     checkpoint_name = f'fine150_{args.checkpoint}'
        #     checkpoint_paths = [output_dir / checkpoint_name]
        #     for checkpoint_path in checkpoint_paths:
        #         save_on_master({
        #             'model': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             # 'model_ema': get_state_dict(model_ema),
        #             'scaler': loss_scaler.state_dict(),
        #             'args': args,
        #         }, checkpoint_path)
                
                
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(data_loader_val)} test images: {test_stats['acc1']:.1f}%")
        
        # print(test_stats["acc1"],"=========")
        # print(epoch >= args.target_epoch)
        
        # if max_accuracy < test_stats["acc1"] and epoch >= args.target_epoch:
        #     max_accuracy = test_stats["acc1"]
        #     if args.output_dir:
        #         checkpoint_name = f'Best_fine150_{args.checkpoint}'
        #         checkpoint_paths = [output_dir / checkpoint_name]
        #         for checkpoint_path in checkpoint_paths:
        #             save_on_master({
        #                 'model': model.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'epoch': epoch,
        #                 'scaler': loss_scaler.state_dict(),
        #                 'args': args,
        #             }, checkpoint_path)
                    
                
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}


    return 0

if __name__ == '__main__':
    args = get_train_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)