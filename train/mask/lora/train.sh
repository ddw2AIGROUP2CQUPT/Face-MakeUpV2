#!/bin/bash

# SDXL训练脚本示例
# 基于train4short_caption_v3.py改造的SDXL版本

accelerate launch -m --num_processes 8 --multi_gpu --mixed_precision bf16 \
    train.mask.lora.train \
    --pretrained_model_name_or_path "benjamin-paine/stable-diffusion-v1-5" \
    --data_json_file "/home/ddwgroup/san/zyx/face_makeup_2/data/caption/HQFaceSquare400WShortCaptionMaskIndex4.json" \
    --data_root_path "/home/ddwgroup/public-Datasets/HQFaceSquare400W/HQFaceSquare400W-1_5/" \
    --face_id_dir "/home/ddwgroup/public-Datasets/HQFaceSquare400W/HQFaceSquare400W_FaceEmbed/FaceCaptionHQ400W_FaceIdEmbed_v1" \
    --control_img_dir "/home/ddwgroup/public-Datasets2/HQFaceSquare400W/V4/Shading" \
    --parsing_mask_dir "/home/ddwgroup/public-Datasets2/HQFaceSquare400W/V4/Mask" \
    --face_img_dir "/home/ddwgroup/public-Datasets/HQFaceSquare400W/HQFaceSquare400W_FaceAlign/" \
    --image_encoder_path "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
    --resolution 512 \
    --train_batch_size 8 \
    --learning_rate 1e-4 \
    --maploss_factor 1e-3 \
    --num_train_epochs 100 \
    --save_steps 20000 \
    --enable_xformers_memory_efficient_attention \
    --dataloader_num_workers 16 \
    --output_dir "checkpoints/train_mask_4face_short_caption_v5" \
    > /home/ddwgroup/san/zyx/face_makeup_2/logs/train/v5.log
    # --resume_from_checkpoint /home/ddwgroup/san/zyx/face_makeup_2/checkpoints/train_mask_4face_short_caption_v4-5/checkpoint-400000 \