export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

############지금 하는건 second step 이후 마스크 생성
############이거렁 마스크 생성 이후 second step도 차이 봐야함. 
#지금 돌리는거 그대로 순서만 바꿔서 돌리기. 지금 돌리는거 성능 확인 후ㅡ 
## 지금 돌리는건 순서 바꿔서 돌리느중 6-26맇 

#3. use dlb sam 사용 x 
for ms in small ; do
   for pi in taylor1 ; do
     for attn_pr in 0.5 ; do
       for ffn_pr in 0.5 ; do
         for mt in grad ; do   # 5
             for mth in unified_qkv ; do
                 for lr in 1e-04 5e-05 ; do
                   for prompt_len in 1 ; do
                     for group_size in 1 ; do
                       for prune_freq in 8 ; do # Added loop for prune-freq with example values  ##10
                            for mask_type in 0  ; do
                                for second_step_after_mask in 1 ; do #1이 원본 e 없는 상태로 mask 생성 
                                    for random_mask_change in 0 ; do
                                            for use_DLB in 1 ; do #에폭 절반  
                                                for DLB_loss in 0 ; do 
                                                    for rho in 0.05 0.0  ; do 
                                                        for overlap in 0 ; do
                                                            for mask_finding_cos in 0 ; do
                                                                for prune_with_e in 0 ; do
                                
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=21888 --use_env cifar.py \
#  --data-set IMNET \
#  --epochs 25 \
#  --data-path /home/20223204/data/CIFAR100 \
#  --gpuids 0 \
#  --cu_num 0 \
#  --model_size ${ms} \
#  --batch-size 256 \
#  --lr ${lr} \
#  --weight-decay 0.05 \
#  --prune-freq ${prune_freq} \
#  --method ${mth} \
#  --warmup 1 \
#  --attn-prune-rate ${attn_pr} \
#  --ffn-prune-rate ${ffn_pr} \
#  --prune-imp ${pi} \
#  --mag_type ${mt} \
#  --target_epoch 25 \
#  --prompt_length ${prompt_len} \
#  --group_size ${group_size} \
#  --rho "${rho}" \
#  --mask_type "${mask_type}" \
#  --second_step_after_mask "${second_step_after_mask}" \
#  --random_mask_change "${random_mask_change}" \
#  --use_DLB "${use_DLB}" \
#  --DLB_loss "${DLB_loss}" \
#  --overlap "${overlap}" \
#  --mask_finding_cos "${mask_finding_cos}" \
#  --prune_with_e "${prune_with_e}" \
#  > ./cifar_log/${ms}_${lr}_attn${attn_pr}_ffn${ffn_pr}_pf${prune_freq}_sam_rho${rho}_use_DLB${use_DLB}_DLB_loss_${DLB_loss}_overlap_${overlap}_mask_finding_cos${mask_finding_cos}_prune_with_e_${prune_with_e}.txt
                                 

# checkpoint_path="cifar100_epoch25_1e-04_small_attn0.5_ffn0.5_8_sam_rho${rho}_use_DLB${use_DLB}_DLB_loss_${DLB_loss}_overlap_${overlap}_mask_finding_cos_${mask_finding_cos}_prune_with_e_${prune_with_e}.pth"
checkpoint_path="cifar100_epoch25_1e-04_small_attn0.5_ffn0.5_8_sam_rho0.05_use_DLB1_DLB_loss_0_overlap_0_mask_finding_cos_0_prune_with_e_0.pth"
                                    checkpoint_filename=$(basename "$checkpoint_path")
                                    checkpoint_basename=${checkpoint_filename%.pth}
log_filename="./fine_log/${ms}_${checkpoint_basename}.txt"
                                    echo "Log file will be saved to: $log_filename"

                                    # --- 파이썬 스크립트 실행 ---
                                    # ${checkpoint_path}가 올바르게 전달되도록 함
                                    python -m torch.distributed.launch --nproc_per_node=1 --master_port=25115 --use_env cifar_fine.py \
                                        --data-set IMNET \
                                        --epochs 75 \
                                        --data-path /home/20223204/data/CIFAR100 \
                                        --checkpoint "${checkpoint_path}" \
                                        --gpuids 0 \
                                        --cu_num 0 \
                                        --model_size "${ms}" \
                                        --batch-size 256 \
                                        --lr "${lr}" \
                                        --weight-decay 0.05 \
                                        --prune-freq 8 \
                                        --method "${mth}" \
                                        --warmup 151 \
                                        --attn-prune-rate "${attn_pr}" \
                                        --ffn-prune-rate "${ffn_pr}" \
                                        --prune-imp "${pi}" \
                                        --mag_type "${mt}" \
                                        --target_epoch -1 \
                                        --group_size "${group_size}" \
                                        --rho "${rho}" \
                                        --mask_type "${mask_type}" \
                                        --second_step_after_mask "${second_step_after_mask}" \
                                         --use_DLB "${use_DLB}" \
                                         --DLB_loss "${DLB_loss}" \
                                         --overlap "${overlap}" \
                                          --mask_finding_cos "${mask_finding_cos}" \
                                        --prune_with_e "${prune_with_e}" \
                                        > "${log_filename}"
                                                        done
                                                     done
                                                   done # Closing the prune_freq loop
                                                 done
                                               done   ###5
                                             done
                                           done
                                       done 
                                   done
                                 done ###10 
                               done
                            done
                        done
                    done
                done
            done
        done
    done
done




export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

############지금 하는건 second step 이후 마스크 생성
############이거렁 마스크 생성 이후 second step도 차이 봐야함. 
#지금 돌리는거 그대로 순서만 바꿔서 돌리기. 지금 돌리는거 성능 확인 후ㅡ 
## 지금 돌리는건 순서 바꿔서 돌리느중 6-26맇 

#3. use dlb sam 사용 x 
for ms in small ; do
   for pi in taylor1 ; do
     for attn_pr in 0.5 ; do
       for ffn_pr in 0.5 ; do
         for mt in grad ; do   # 5
             for mth in unified_qkv ; do
                 for lr in 1e-04 5e-05 ; do
                   for prompt_len in 1 ; do
                     for group_size in 1 ; do
                       for prune_freq in 8 ; do # Added loop for prune-freq with example values  ##10
                            for mask_type in 0  ; do
                                for second_step_after_mask in 1 ; do #1이 원본 e 없는 상태로 mask 생성 
                                    for random_mask_change in 0 ; do
                                            for use_DLB in 0 ; do #에폭 절반  
                                                for DLB_loss in 0 ; do 
                                                    for rho in 0.05 0.0  ; do 
                                                        for overlap in 0 ; do
                                                            for mask_finding_cos in 0 ; do
                                                                for prune_with_e in 0 ; do
                                       

checkpoint_path="cifar100_epoch25_1e-04_small_attn0.5_ffn0.5_8_sam_rho0.05_use_DLB1_DLB_loss_0_overlap_0_mask_finding_cos_0_prune_with_e_0.pth"
                                    checkpoint_filename=$(basename "$checkpoint_path")
                                    checkpoint_basename=${checkpoint_filename%.pth}
log_filename="./fine_log/${ms}_${checkpoint_basename}.txt"
                                    echo "Log file will be saved to: $log_filename"

                                    # --- 파이썬 스크립트 실행 ---
                                    # ${checkpoint_path}가 올바르게 전달되도록 함
                                    python -m torch.distributed.launch --nproc_per_node=2 --master_port=25115 --use_env cifar_fine.py \
                                        --data-set IMNET \
                                        --epochs 150 \
                                        --data-path /home/20223204/data/CIFAR100 \
                                        --checkpoint "${checkpoint_path}" \
                                        --gpuids 0 1 \
                                        --cu_num 0,1 \
                                        --model_size "${ms}" \
                                        --batch-size 512 \
                                        --lr "${lr}" \
                                        --weight-decay 0.05 \
                                        --prune-freq 8 \
                                        --method "${mth}" \
                                        --warmup 151 \
                                        --attn-prune-rate "${attn_pr}" \
                                        --ffn-prune-rate "${ffn_pr}" \
                                        --prune-imp "${pi}" \
                                        --mag_type "${mt}" \
                                        --target_epoch -1 \
                                        --group_size "${group_size}" \
                                        --rho "${rho}" \
                                        --mask_type "${mask_type}" \
                                        --second_step_after_mask "${second_step_after_mask}" \
                                         --use_DLB "${use_DLB}" \
                                         --DLB_loss "${DLB_loss}" \
                                         --overlap "${overlap}" \
                                          --mask_finding_cos "${mask_finding_cos}" \
                                        --prune_with_e "${prune_with_e}" \
                                        > "${log_filename}"
                                                        done
                                                     done
                                                   done # Closing the prune_freq loop
                                                 done
                                               done   ###5
                                             done
                                           done
                                       done 
                                   done
                                 done ###10 
                               done
                            done
                        done
                    done
                done
            done
        done
    done
done


