export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="image_folder/blue_illu_folder"
export OUTPUT_DIR="lora_output/blue_illu_5e-5_NoTE_1000steps_long_long"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export PRETRAINED_LORA_WEIGHT_DIR="/mnt/Face_Private-NFS2/geonhui/works2_ckpt/blue_illu_5e-5_NoTE_1000steps_revised_no_color.safetensors"

CUDA_VISIBLE_DEVICES="GPU-e7718400-498b-9fe1-4888-1ff1f272a7dc" python train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a woman working on laptop with blue plants and shelf on a white background in flat illustration style, minimal simple vector graphics" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5 \
  --validation_prompt="a woman working on laptop with blue plants and shelf on a white background in flat illustration style, minimal simple vector graphics" \
  --validation_epochs=7577 \
  --seed="0" \
  --rank=64 \
  --pretrained_lora_weight_dir=$PRETRAINED_LORA_WEIGHT_DIR



