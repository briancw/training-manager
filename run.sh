#!/bin/bash

OUTPUT_NAME="name-of-the-new-model"
INSTANCE_PROMPT="xyz person"
MODEL_NAME="sd-v1-4"
TRAINING_IMAGES="person-v1"
EPOCHS=15
SAVE_EPOCHS=10

# Define paths to wherever you keep your models and images
MODELS_BASE_PATH="/mnt/sd/diffusers-models"
TRAINING_BASE_PATH="/mnt/sd/training-images"

# Generated Paths
MODEL_PATH="$MODELS_BASE_PATH/$MODEL_NAME"
TRAINING_PATH="$TRAINING_BASE_PATH/$TRAINING_IMAGES"
OUTPUT_PATH="$MODELS_BASE_PATH/$OUTPUT_NAME"

# Activate virtualenv
source venv/bin/activate/

# Train
python train.py \
  --instance_prompt="$INSTANCE_PROMPT" \
  --model_path=$MODEL_PATH \
  --instance_data_dir=$TRAINING_PATH \
  --output_dir=$OUTPUT_PATH \
  --seed=80085 \
  --resolution=512 \
  --center_crop \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --sample_batch_size=4 \
  --num_train_epochs=$EPOCHS \
  --save_each_epoch=$SAVE_EPOCHS