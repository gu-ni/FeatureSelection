export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export INSTANCE_DIR="image_folder/bell_illustration_folder"
# export OUTPUT_DIR="lora-trained-xl_bell_illustration_5e-5_TE"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

INSTANCE_DIR=(
  "image_folder/dog_folder" \
  "image_folder/blue_illu_folder" \
  "image_folder/flower_watercolor_folder" \
  "image_folder/watermelon_3d_folder" \
  "image_folder/tree_sticker_folder" \
  "image_folder/fox_watercolor_foder" \
  "image_folder/matt_black_sculpture_folder" \
  "image_folder/golden_temples_folder" \
  "image_folder/chauchau_folder"
)

OUTPUT_DIR=(
  "output_dir_64rank/NoTE/dog_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/blue_illu_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/flower_watercolor_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/watermelon_3d_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/tree_sticker_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/fox_watercolor_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/matt_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/golden_temples_5e-5_NoTE_1000steps" \
  "output_dir_64rank/NoTE/chauchau_5e-5_NoTE_1000steps"
)

InstancePrompt=(
  "a photo of ohwx dog" \
  "a woman working on a laptop in flat cartoon illustration style" \
  "flowers in watercolor painting style" \
  "slices of watermelon and clouds in the background in 3d rendering style" \
  "a flat illustration style sticker of Christmas tree" \
  "a fox in watercolor painting style" \
  "a matt black sculpture of gorilla face" \
  "a realistic landscape painting of golden temples in a colorful forest" \
  "a photo of szn dog"
)

for i in "${!INSTANCE_DIR[@]}"; do 
  CUDA_VISIBLE_DEVICES="MIG-0b2452d4-9b27-530f-a6f1-1c2d05dfaa72" python train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir="${INSTANCE_DIR[$i]}" \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir="${OUTPUT_DIR[$i]}" \
  --mixed_precision="fp16" \
  --instance_prompt="${InstancePrompt[$i]}" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="${InstancePrompt[$i]}" \
  --validation_epochs=7577 \
  --seed="0" \
  --rank=64
done
