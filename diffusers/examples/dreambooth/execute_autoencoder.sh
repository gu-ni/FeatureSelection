export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="image_folder/blue_illu_folder"
export OUTPUT_DIR="lora_output/blue_illu_5e-5_NoTE_1000steps_SwappingAE_test"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CUDA_VISIBLE_DEVICES="GPU-68b77852-bf59-f9d6-dad1-09a92e44c6f4" python train_dreambooth_lora_sdxl_guni.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a woman in minimal simple illustration, vector graphics" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5 \
  --validation_prompt="a woman in minimal simple illustration, vector graphics" \
  --validation_epochs=7577 \
  --seed="0" \
  --rank=64 \
  --pretrained_lora_weight_dir="" \
  --train_swapping_autoencoder=True \
  --swapping_autoencoder_weight=1.0 \
  --upscaled_prompt="a woman working on laptop with blue plants and shelf on a white background in flat illustration style, minimal simple vector graphics"


