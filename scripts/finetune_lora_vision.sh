#!/bin/bash

MODEL_NAME="mistral-community/pixtral-12b"

# Pixtral does not support flash-attnetion2 yet.
# The multi-modal projector isn't included in the lora module, you should set tune_img_projector to True.
# Also it could be better for setting the lr for img_procjector.

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --disable_flash_attn2 True \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bf16 True \
    --output_dir output/lora_vision_test1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \