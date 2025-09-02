export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0,lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1



for ms in small ; do
    for pi in taylor1 ; do
      for attn_pr in 0.35 ; do
          for ffn_pr in 0.45 ; do
              for mt in grad ; do
                  for mth in ours ; do
                      for lr in 5e-5 ; do
                        for group_size in 16 ; do
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29511 --use_env fine.py \
 --data-set IMNET \
 --epochs 150 \
 --data-path /workspace/data/ImageNet \
 --checkpoint ./output/timm_pretrain50_taylor_0.0002small_attn0.3_ffn0.4_ours_weight_L18_imagnet.pth \
 --gpuids 0 1 2 3 \
 --cu_num 0,1,2,3 \
 --model_size ${ms} \
 --batch-size 512 \
 --lr ${lr} \
 --weight-decay 0.05 \
 --prune-freq 8 \
 --method ${mth} \
 --warmup 151 \
 --attn-prune-rate ${attn_pr} \
 --ffn-prune-rate ${ffn_pr} \
 --prune-imp ${pi} \
 --mag_type ${mt} \
 --target_epoch -1 \
 --group_size ${group_size} \
 > ./fine_log/${ms}_taytm1_pre50_fine150_${lr}_attn${attn_pr}_ffn${ffn_pr}_gs${group_size}_${mth}_${mt}_${pi}_imagenet.txt
                        done
                      done
                    done
                done 
            done
        done
    done
done