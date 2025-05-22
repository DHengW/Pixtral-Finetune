#!/bin/bash

# 修改为您的模型路径
MODEL_NAME="/root/autodl-tmp/mistral-community/pixtral-12b"

# Pixtral does not support flash-attnetion2 yet.
# It only supports batch size 1 for now. If you want to use batch size > 1, you need to modify the model code. The model dose not support various image sizes
# in the same batch. If you want to use various image sizes, you need to modify the model code.

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# 多机多卡训练的NCCL通信优化
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BUFFSIZE=2097152

# 下面的参数需要根据您的集群设置进行调整
MASTER_ADDR="localhost" # 主节点IP地址
MASTER_PORT="29500"     # 主节点端口
NUM_NODES=2             # 节点数量
NUM_GPUS_PER_NODE=4     # 每个节点的GPU数量
NODE_RANK=0             # 当前节点的排名，主节点为0，其他节点依次递增

# 导出分布式训练需要的环境变量
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# 为多节点训练准备hostfile
echo "创建hostfile..."
cat > hostfile << EOF
$MASTER_ADDR slots=$NUM_GPUS_PER_NODE
# 如有其他节点，请在下方添加，格式如下
# node2_ip slots=$NUM_GPUS_PER_NODE
# node3_ip slots=$NUM_GPUS_PER_NODE
EOF

# 使用deepspeed启动多节点训练
deepspeed --hostfile hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /root/autodl-tmp/kdd_final_index/aug_dataset/s1_dataset/s1_v1.json \
    --image_folder /root/autodl-tmp/kdd_final_index/aug_dataset/s1_dataset/ \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --fp16 True \
    --bf16 False \
    --output_dir /root/autodl-tmp/Pixtral-Finetune/output/full_finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 2 \
    --save_steps 500 \
    --save_total_limit 2