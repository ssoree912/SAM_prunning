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
import wandb
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


import transformers
from torch.nn.parallel import DistributedDataParallel as DDP

# ===== 여기서부터 추가하거나 기존 코드에 삽입 =====
print("--- Debug Info from cifar.py ---")
print(f"Python Executable: {sys.executable}")
if 'VIRTUAL_ENV' in os.environ:
    print(f"Virtual Environment: {os.environ['VIRTUAL_ENV']}")
else:
    print("Virtual Environment: Not detected")
try:
    import torch
    print(f"PyTorch Version (inside cifar.py): {torch.__version__}")
    print(f"PyTorch Path (inside cifar.py): {torch.__path__}")
except ImportError:
    print("PyTorch not found inside cifar.py")

try:
    import timm
    print(f"Timm Version (inside cifar.py): {timm.__version__}")
    print(f"Timm Path (inside cifar.py): {timm.__path__}")
except ImportError:
    print("Timm not found inside cifar.py")

try:
    import huggingface_hub
    print(f"Hugging Face Hub Version (inside cifar.py): {huggingface_hub.__version__}")
    print(f"Hugging Face Hub Path (inside cifar.py): {huggingface_hub.__path__}")
except ImportError:
    print("Hugging Face Hub not found inside cifar.py")
print("--------------------------------")
# ===== 여기까지 추가하거나 기존 코드에 삽입 =====



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
from MaskedSAM import MaskedSAM

# === Helper Functions/Classes (from Script 1) ===

def replace_all_linear_layers(module):
    """
    Recursively replaces all nn.Linear layers in a mosdule with MaskLinear layers.
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

# Note: CustomScheduler and reset_momentum from script 1 are not used in the main flow provided.
# They can be added here if needed.

# === Main Training Function ===

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
    checkpoint_path = model_configs[args.model_size]['checkpoint']

    print(f"Creating model: {model_name}")
    # Create model with CIFAR-100 number of classes
    model = timm.create_model(model_name, pretrained=True, num_classes=100)

    # Load pre-trained weights (if checkpoint exists)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        # Use map_location='cpu' to avoid GPU memory issues when loading
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get the state dict from the checkpoint
        # Adjust the key ('model') if your checkpoint structure is different
        checkpoint_state_dict = checkpoint['model']

        # --- Start Modification ---
        # Get the state dict of the current model (with 100 classes)
        model_state_dict = model.state_dict()

        # Keys for the classification head to remove from the checkpoint
        keys_to_remove = ['head.weight', 'head.bias']
        removed_keys = []

        # Filter out the incompatible head keys from the checkpoint state_dict
        filtered_checkpoint_state_dict = {}
        for k, v in checkpoint_state_dict.items():
            if k in model_state_dict and k not in keys_to_remove and model_state_dict[k].shape == v.shape:
                 filtered_checkpoint_state_dict[k] = v
            elif k in keys_to_remove:
                 removed_keys.append(k)
            # Optional: You might want to log other mismatches if they occur
            # else:
            #    print(f"Skipping key {k}: Shape mismatch or key not in model.")


        if removed_keys:
             print(f"Removed classification head keys from checkpoint: {removed_keys}")
        # --- End Modification ---


        # Load the filtered state dict. strict=False handles missing keys (like the head).
        # Use the filtered dictionary here
        load_result = model.load_state_dict(filtered_checkpoint_state_dict, strict=False)
        print("Checkpoint loading report (missing keys should include head weights):")
        print("  Missing keys:", load_result.missing_keys)
        print("  Unexpected keys:", load_result.unexpected_keys) # Should ideally be empty

    else:
        print(f"Checkpoint not found at {checkpoint_path}. Training from scratch or using timm's pretrained.")
        # Optionally load timm's pretrained weights if checkpoint is missing
        # model = timm.create_model(model_name, pretrained=True, num_classes=args.nb_classes)

    # --- Apply Pruning Layers and Modifications ---
    # (Rest of the code remains the same)
    print("Replacing Linear layers with MaskLinear")
    replace_all_linear_layers(model)
    print("Setting Dropout to 0.0")
    set_dropout_to_zero(model)

    # Wrap model
    model = VitWrapper(model)
    model.to(device)
    model.set_all_type_values(0)
    print("완료 ㅋㅋ")

    # Print model structure
    # print(model)

    # --- DistributedDataParallel / DataParallel ---
    if args.distributed:
        # 분산 학습일 경우: DDP로 감싸고 .module로 원본 모델 접근
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        # 분산 학습이 아닐 경우: 모델 자체가 원본 모델
        model_without_ddp = model

    # 모델 타입 설정은 조건문 밖에서 한 번만 호출
    # (model_without_ddp는 어떤 경우든 원본 모델을 가리킴)
    model_without_ddp.set_all_type_values(0)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable params:', n_parameters)

    # --- Optimizer ---
    orin_lr = args.lr
    # Scale learning rate based on global batch size (from script 1)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr  * get_world_size() 
        args.lr = linear_scaled_lr
        print(f"Scaled learning rate to: {args.lr}")

    # Use AdamW optimizer (same as script 1)
    # optimizer = AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("args.second_step_after_mask")
    print(args.second_step_after_mask)
    rho = args.rho
    print("SAM rho 지정 바로 아래줄")
    print(rho)
    print()
    
    optimizer = MaskedSAM(model.parameters(), 
                    AdamW, 
                    rho=rho,                     # SAM neighbourhood radius
                    adaptive=False,               # ASAM 여부
                    v2=True,                       # one-step SAM(v2) 여부
                    lr=args.lr,                           # AdamW 에 들어갈 lr
                    betas=(0.9, 0.999),                   # AdamW 전용 인자
                    eps=1e-8,
                    weight_decay=args.weight_decay,       # AdamW decoupled WD
                   )
                       

                    

    # --- Loss Scaler (for AMP) ---
    loss_scaler = NativeScaler()

    # --- LR Scheduler ---
    # Use CosineAnnealingLR (same as script 1)
    # Note: Script 1 didn't use CosineAnnealingWarmRestarts in the main flow, using CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    print("스케줄러: , ", lr_scheduler)
    # --- Loss Function (Criterion) ---
    # Select loss function based on arguments (from script 1)
  
    criterion = torch.nn.CrossEntropyLoss()

    # --- Model EMA Setup (from Script 1) ---
    model_ema = None


    # --- Checkpoint/Output Setup ---
    output_dir = Path(args.output_dir)
    # Create output directory if it doesn't exist (only on master process)
    if args.output_dir and is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume Logic (Optional - Add if needed) ---
    # if args.resume:
    #     # Load checkpoint logic similar to script 1
    #     pass

    # --- Pruning State Initialization ---
    iterations = 0
    mask_changes = 0
    prev_masks = {}
    # Initialize previous masks for tracking changes (only needed if prune_imp includes 'mask_change')
    # Use model_without_ddp to access modules directly
    print("Initializing previous masks...")
    for name, module in model_without_ddp.named_modules():
       if isinstance(module, MaskLinear):
           # Initialize prev_masks with ones (assuming initially no pruning)
           prev_masks[name] = torch.ones_like(module.mask, device=module.mask.device)
           print(f"  Initialized prev_mask for: {name}")


    # === Training Loop ===
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    # if args.use_DLB== 1: 
    #     model_without_ddp = extend(model_without_ddp)
    #     criterion = extend(criterion)
    #     gpu_id_list = args.gpuids.split(',') # ['0', '1'] 리스트로 분리

    #     # 현재 프로세스의 순번(rank)에 맞는 GPU ID를 선택
    #     # args.cu_num이 현재 프로세스의 순번(0, 1, ...)이라고 가정
    #     current_gpu_id = int(gpu_id_list[int(args.cu_num)])
    #     local_rank_device_ids = [current_gpu_id]

    #     model = DDP(model_without_ddp, device_ids=local_rank_device_ids, output_device=current_gpu_id)

    
    for epoch in range(args.epochs):
        if args.distributed:
            # Set epoch for distributed sampler to ensure proper shuffling
            data_loader_train.sampler.set_epoch(epoch)

        # --- Train One Epoch ---
        # Call train_one_epoch (ensure signature matches script 1's usage)
        model_without_ddp.set_all_type_values(args.mask_type)
        print(f"train 전에 set mask type{args.mask_type} 변경")
        
#         1. 다른거 없이 sam v1만 == train_one_epoch_sam_origin (에폭 그대로
# 2. sam v1에 dlb 적용 ==train_one_epoch_dlb_sam_origin (에폭 절반 
# 3. sam v1 에 dlb 하는데 겹치는 배치만 e (sam 보다 계산량 25퍼 줄어듦) == train_one_epoch_DLB (에폭 절반
        
    
    
        if args.use_DLB== 0: 
            train_stats, attn_target_sparsity, ffn_target_sparsity, iterations , prev_masks, mask_changes, zero_metrics = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.train_mode, # or False if not used
                args=args, # Pass args for pruning logic inside train_one_epoch
                iteration=iterations, # Pass current iteration count
                prev_masks=prev_masks, # Pass previous masks
                mask_changes=mask_changes # Pass mask change count
            )

        elif args.use_DLB== 1: 

            train_stats, attn_target_sparsity, ffn_target_sparsity, iterations , prev_masks, mask_changes, zero_metrics = train_one_epoch_DLB_test(
                model,criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.train_mode, # or False if not used
                args=args, # Pass args for pruning logic inside train_one_epoch
                iteration=iterations, # Pass current iteration count
                prev_masks=prev_masks, # Pass previous masks
                mask_changes=mask_changes # Pass mask change count
            )                                                                           
                                                                
        # --- LR Scheduler Step ---
        lr_scheduler.step() # Step scheduler after epoch

        # --- Logging ---
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'  Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}') # Get LR from optimizer
        # Print sparsity (names might differ based on train_one_epoch implementation)
        print(f'  Attention Sparsity Target: {attn_target_sparsity}')
        print(f'  FFN Sparsity Target: {ffn_target_sparsity}')
        print(f'  Mask Changes: {mask_changes}')


        # --- Evaluation ---
        # Run evaluation on the validation set
        test_stats = evaluate(data_loader_val, model, device)
        print(f"  Accuracy on CIFAR-100 test set: {test_stats['acc1']:.2f}%")

        # Update max accuracy
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        
        # Log epoch and max accuracy to wandb
        if args.wandb:
            import wandb
            log_dict = {
                "epoch": epoch,
                "test/accuracy": test_stats["acc1"],
                "test/acc5": test_stats["acc5"],
                "test/loss": test_stats["loss"],
                "test/max_accuracy": max_accuracy,
                "train/loss": train_stats.get("loss", 0),
                "train/lr": train_stats.get("lr", 0),
                "train/attn_sparsity": attn_target_sparsity,
                "train/ffn_sparsity": ffn_target_sparsity,
                "train/mask_changes": mask_changes
            }
            
            # Add zero metrics if available
            if zero_metrics is not None:
                log_dict.update({
                    "pruning/total_weights_zero": zero_metrics['total_weights_zero'],
                    "pruning/total_grads_zero": zero_metrics['total_grads_zero'],
                    "pruning/total_masked_weights_zero": zero_metrics['total_masked_weights_zero'],
                    "pruning/total_masked_grads_zero": zero_metrics['total_masked_grads_zero']
                })
            
            wandb.log(log_dict)
        print(f'  Max accuracy so far: {max_accuracy:.2f}%')

        # --- Checkpoint Saving ---
        # Save checkpoint periodically or based on performance
        lr_for_filename = f'{orin_lr:.0e}'
        if args.output_dir and (epoch + 1) == args.target_epoch:  # Example: Save every 'save_freq' epochs
            checkpoint_name = f'cifar100_epoch{args.epochs}_{lr_for_filename}_{args.model_size}_attn{str(args.attn_prune_rate)}_ffn{str(args.ffn_prune_rate)}_{args.prune_freq}_sam_rho{args.rho}_use_DLB{args.use_DLB}_DLB_loss_{args.DLB_loss}_overlap_{args.overlap}_mask_finding_cos_{args.mask_finding_cos}_prune_with_e_{args.prune_with_e}.pth'
           

            checkpoint_path = output_dir / checkpoint_name
            print(f"Saving checkpoint to {checkpoint_path}")
            save_on_master({
                # Save model state (use model_without_ddp for non-distributed saving)
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema) if model_ema else None,  # Save EMA state if used
                'scaler': loss_scaler.state_dict(),  # Save AMP scaler state
                'args': args,  # Save arguments used for this run
                'max_accuracy': max_accuracy,  # Save best accuracy achieved
            }, checkpoint_path)




        # --- Log Stats (JSON) ---
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters,
                     'learning_rate': optimizer.param_groups[0]["lr"],
                     'attn_sparsity': attn_target_sparsity, # Log sparsity
                     'ffn_sparsity': ffn_target_sparsity}


        # if args.output_dir and is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    # === End of Training ===
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# === Script Entry Point ===
if __name__ == '__main__':
    args = get_train_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)