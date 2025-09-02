export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

for ms in small ; do
   for pi in L1  ; do
     for pr in 0.5 ; do
         for mt in grad ; do
             for mth in nfor ; do
                 for lr in 5e-4 ; do
                   for prompt_len in 1 10 50 100 ; do
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29679 --use_env cifar.py \
 --data-set IMNET \
 --epochs 50 \
 --data-path /workspace/data/ImageNet \
 --g 0  \
 --cu_num 3 \
 --model_size ${ms} \
 --batch-size 512 \
 --lr ${lr} \
 --weight-decay 0.05 \
 --prune-freq 8 \
 --method ${mth} \
 --warmup 1 \
 --prune-rate ${pr} \
 --prune-imp ${pi} \
 --mag_type ${mt} \
 --target_epoch 50 \
 --prompt_length ${prompt_len} \
 > ./cifar_log/${ms}_token_add_pretraind_${lr}_${pr}_${mth}_${mt}_${pi}_pl${prompt_len}_cifar.txt
                     done
                   done
                 done
               done
           done 
       done
   done
