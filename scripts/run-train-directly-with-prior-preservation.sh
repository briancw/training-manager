#!/bin/bash

# You may need to specify a path to "libcudart.so"
# On arch linux uncomment the following line
# export LD_LIBRARY_PATH=/opt/cuda/targets/x86_64-linux/lib/

OUTPUT_NAME="name-of-the-new-model"
INSTANCE_PROMPT="xyz person"
CLASS_PROMPT="person"
MODEL_NAME="sd-v1-4"
TRAINING_IMAGES="person"
CLASS_IMAGES="person-images"
#CLASS_IMAGE_COUNT=200
EPOCHS=10
SAVE_EPOCHS=2

# Define paths to wherever you keep your models and images
MODELS_BASE_PATH="/mnt/sd/diffusers-models"
TRAINING_BASE_PATH="/mnt/sd/training-images"
CLASS_BASE_PATH="/mnt/sd/class-images"

# Generated Paths
MODEL_PATH="$MODELS_BASE_PATH/$MODEL_NAME"
TRAINING_PATH="$TRAINING_BASE_PATH/$TRAINING_IMAGES"
CLASS_PATH="$CLASS_BASE_PATH/$CLASS_IMAGES"
OUTPUT_PATH="$MODELS_BASE_PATH/$OUTPUT_NAME"

# Activate virtualenv
source ./venv/bin/activate

# Train
python train.py \
  --instance_prompt="$INSTANCE_PROMPT" \
  --class_prompt="$CLASS_PROMPT" \
  --model_path=$MODEL_PATH \
  --instance_data_dir=$TRAINING_PATH \
  --class_data_dir=$CLASS_PATH \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --output_dir=$OUTPUT_PATH \
  --seed=80085 \
  --resolution=512 \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --sample_batch_size=4 \
  --num_train_epochs=$EPOCHS \
  --save_each_epoch=$SAVE_EPOCHS
