export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

for ms in small ; do
   for pi in taylor1 ; do
     for attn_pr in 0.3 ; do
       for ffn_pr in 0.4 ; do
         for mt in grad ; do
             for mth in ours ; do
                 for lr in 5e-5 ; do
                   for prompt_len in 1 ; do
                     for group_size in 16 ; do
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29573 --use_env exploration.py \
 --data-set IMNET \
 --epochs 50 \
 --data-path /workspace/data/ImageNet \
 --gpuids 0 1 2 3 \
 --cu_num 0,1,2,3 \
 --model_size ${ms} \
 --batch-size 512 \
 --lr ${lr} \
 --weight-decay 0.05 \
 --prune-freq 8 \
 --method ${mth} \
 --warmup 1 \
 --attn-prune-rate ${attn_pr} \
 --ffn-prune-rate ${ffn_pr} \
 --prune-imp ${pi} \
 --mag_type ${mt} \
 --target_epoch 50 \
 --prompt_length ${prompt_len} \
 --group_size ${group_size} \
 > ./log/${ms}_taytm1_pretraind_${lr}_attn${attn_pr}_ffn${ffn_pr}_gs${group_size}_${mth}_${mt}_${pi}_pl${prompt_len}_imagenet.txt
                     done
                   done
                 done
               done
           done 
       done
     done
   done
done


# for ms in small ; do
#    for pi in L1 ; do
#      for attn_pr in 0.6 ; do
#        for ffn_pr in 0.8 ; do
#          for mt in weight ; do
#              for mth in neuron ; do
#                  for lr in 5e-5 ; do
#                    for prompt_len in 1 ; do
#                      for group_size in 16 ; do
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=29512 --use_env exploration.py \
#  --data-set IMNET \
#  --epochs 1 \
#  --data-path /workspace/data/ImageNet \
#  --gpuids 0 1 2 3 \
#  --cu_num 0,1,2,3 \
#  --model_size ${ms} \
#  --batch-size 512 \
#  --lr ${lr} \
#  --weight-decay 0.05 \
#  --prune-freq 8 \
#  --method ${mth} \
#  --warmup -1 \
#  --attn-prune-rate ${attn_pr} \
#  --ffn-prune-rate ${ffn_pr} \
#  --prune-imp ${pi} \
#  --mag_type ${mt} \
#  --target_epoch 1 \
#  --prompt_length ${prompt_len} \
#  --group_size ${group_size} \
#  > ./log/${ms}_taytm1_pretraind_${lr}_attn${attn_pr}_ffn${ffn_pr}_gs${group_size}_${mth}_${mt}_${pi}_pl${prompt_len}_imagenet.txt
#                      done
#                    done
#                  done
#                done
#            done 
#        done
#      done
#    done
# done




