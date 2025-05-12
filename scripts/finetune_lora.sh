#!/bin/bash

MODEL_NAME="data/mistral-community/pixtral-12b"

# Pixtral does not support flash-attnetion2 yet.

# Pixtral does not support flash-attnetion2 yet.
# It only supports batch size 1 for now. If you want to use batch size > 1, you need to modify the model code. The model dose not support various image sizes
# in the same batch. If you want to use various image sizes, you need to modify the model code.

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora False \
    --use_dora False \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path data/teamix-aicrowd/index_aug_image_rerank_dataset/data_json.json \
    --image_folder data/teamix-aicrowd/index_aug_image_rerank_dataset/path/ \
    --disable_flash_attn2 True \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --output_dir output/test_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
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
    --save_steps 500 \
    --save_total_limit 10