export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="image_folder/blue_illu_folder"
export OUTPUT_DIR="64rank_dreambooth_output/blue_illu_5e-5_NoTE_1000steps_long_prompt"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

CUDA_VISIBLE_DEVICES="GPU-48d0547b-e505-7f76-7741-a012f5c16f7a" python train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a woman surrounded by blue decoration on a white background in flat cartoon illustration style" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="a woman surrounded by blue decoration on a white background in flat cartoon illustration style" \
  --validation_epochs=7577 \
  --seed="0" \
  --rank=64



