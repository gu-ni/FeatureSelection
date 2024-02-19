# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-48d0547b-e505-7f76-7741-a012f5c16f7a"
import torch
from diffusers import DiffusionPipeline
from PIL import Image

pipe = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16")
pipe = pipe.to("cuda")

# %%
#pipe.load_lora_weights("64rank_ziplora_output/dog_and_outback_painting/dog_5e-5_NoTE_1000steps_zip_1e-3_100steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="dog")
#pipe.load_lora_weights("64rank_ziplora_output/dog_and_outback_painting/outback_painting_5e-5_NoTE_1000steps_zip_1e-3_100steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="outback_painting")

#pipe.load_lora_weights("64rank_dreambooth_output/dog_5e-5_NoTE_1000steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="dog")
pipe.load_lora_weights("64rank_dreambooth_output/blue_illu_5e-5_NoTE_1000steps_long_prompt", weight_name="pytorch_lora_weights.safetensors", adapter_name="blue_illu")
# %%
pipe.set_adapters(["dog", "outback_painting"], adapter_weights=[1.0, 1.0])
# %%
pipe.set_adapters(["dog", "outback_painting"], adapter_weights=[1.0, 1.0])
# %% separated
seeds = range(5)
result_image = Image.new('RGB', (1024 * len(seeds), 1024), color='white')
#prompt = "a ohwx dog in pastel painting style"
#prompt = "a ohwx dog in flat cartoon illustration style"
prompt = "a photo of a ohwx dog in watercolor painting style"

for i in seeds:
    generator = torch.Generator("cuda").manual_seed(i)
    image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    result_image.paste(image, (i * 1024, 0))

result_image.show()

# %%

prompts = [
    "playing with a ball",
    "catching a frisbie",
    "wearing a hat",
    "with a crown",
    "riding a bicycle",
    "sleeping",
    "in a boat",
    "driving a car"
]
object = "a ohwx dog"
desc = "in pastel painting style"
#desc = "in flat cartoon illustration style"
#desc = "in watercolor painting style"
prompts = [f'{object} {p} {desc}' for p in prompts]
#prompts = [f'{object} {p}' for p in prompts]
# %%
descs = [
    "in pastel painting style",
    "in flat cartoon illustration style",
    "in watercolor painting style",
    ", macro photo, 3d game asset",
    "in 3d rendering style"
]

# %%


result_image = Image.new('RGB', (1024 * len(prompts), 1024), color='white')
for i, prompt in enumerate(prompts):
    generator = torch.Generator("cuda").manual_seed(5)
    image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    result_image.paste(image, (i * 1024, 0))

result_image.show()
    
# %%
prompts = [
    "a ohwx dog, flat cartoon illustration style",
    "a ohwx dog, 3d rendering style",
    "a ohwx dog, macro photo, 3d game asset",
    "a ohwx dog, watercolor painting style",
    "a ohwx dog, medieval painting",
]

for seed in [7, 77, 777]:
    result_image = Image.new('RGB', (1024 * len(prompts), 1024), color='white')
    for i, prompt in enumerate(prompts):
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
        result_image.paste(image, (i * 1024, 0))

    result_image.show()
# %%

for seed in range(5):
    prompt = "a ohwx dog in watercolor painting style"
    image = pipe(
        prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(seed)
    ).images[0]
    image.show()
# %%

# %%
pipe.load_lora_weights('output_dir_64rank/NoTE/flower_watercolor_5e-5_NoTE_1000steps')
#pipe.load_lora_weights('output_dir_TE/lora-trained-xl_matt_5e-5_TE')

# %%
prompts = [
    'a matt black sculpture of gorilla face',
    'a gorilla face',
    'a matt black sculpture',
    'a matt black sculpture of dog face',
    'a matt black sculpture of an old woman with earrings',
    'a matt black sculpture of an old man with eyeglasses and beard',
    'a matt black sculpture of a house',
    'a matt black sculpture of an American street cat in bathtub'
]

prompts = [
    'a woman working on a laptop in flat cartoon illustration style',
    'a dog in flat cartoon illustration style',
    'an orange in flat cartoon illustration style',
    'an avocado in flat cartoon illustration style'
]

prompts = [
    'slices of watermelon and clouds in the background in 3d rendering style',
    'a dog in 3d rendering style',
    'an orange in 3d rendering style',
    'an avocado in 3d rendering style',
]

# %%
prompts = [
    'a woman surrounded by blue decoration on a white background in flat cartoon illustration style',
    'a dog surrounded by blue decoration on a white background in flat cartoon illustration style',
    'an orange surrounded by blue decoration on a white background in flat cartoon illustration style',
    'an avocado surrounded by blue decoration on a white background in flat cartoon illustration style',
    'a man surrounded by blue decoration on a white background in flat cartoon illustration style'
]

result_image = Image.new('RGB', (1024 * len(prompts), 1024), color='white')

for i, prompt in enumerate(prompts):
    generator = torch.Generator("cuda").manual_seed(111)
    print(prompt)
    image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    result_image.paste(image, (i * 1024, 0))

result_image.show()

# %%
# [3, 33, 333, 3333, 33333, 333333, 3333333]
prompt = 'a man surrounded by blue decoration on a white background in flat cartoon illustration style'
for seed in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
    print(seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
    image.show()
# %% get_prompts 정의
prefix = [
    'a dog', 'an orange', 'clouds', 'slices of watermelon', 'a man', 'a baby penguin', 
    'a moose', 'a towel', 'an espresso machine', 'an avocado', 'a crown', 
    'a fluffy baby sloth with a knitted hat trying to figure out a laptop, close up', 
    'the Golden Gate bridge', 'a man riding a snowboard']
object_dict = {
    'woman_illu': 'a woman walking a dog',
    'cat_watercolor': 'a cat',
    'blue_illu': 'a woman',
    'flower_watercolor': 'flowers', 
    'watermelon_3d': 'slices of watermelon and clouds in the background',
    'tree_sticker': 'Christmas tree',
    'fox_watercolor': 'a fox',
    'outback_painting': 'an open landscape with trees',
    'bell_illu': 'a bell',
    'golden_temples': 'golden temples'        
}
def get_prompts(key):
    prefix_ = [object_dict[key]] + prefix
    prompts_dict = {
        'woman_illu':
            [f'{x} in flat cartoon illustration style' for x in prefix_],
        'cat_watercolor':
            [f'{x} in watercolor painting style' for x in prefix_],        
        'blue_illu':
            [f'{x} with decorations on a white background in flat cartoon illustration style' for x in prefix_],
        'flower_watercolor':
            [f'{x} in watercolor painting style' for x in prefix_],
        'watermelon_3d':
            [f'{x} in 3d rendering style' for x in prefix_],
        'tree_sticker':
            [f'a flat illustration style sticker of {x}' for x in prefix_],
        'fox_watercolor':
            [f'{x} in watercolor painting style' for x in prefix_],
        'outback_painting':
            [f'{x} in pastel painting style' for x in prefix_],
        'bell_illu':
            [f'{x} in flat vector line illustration style' for x in prefix_],
        'golden_temples':
            [f'a realistic landscape painting of {x} in a colorful forest' for x in prefix_]
    }
    return prompts_dict[key]
# %%
cwd = os.getcwd()
output_dir = os.path.join(cwd, 'output_dir_64rank/NoTE')
output_files = os.listdir(output_dir)

for file in output_files:
    for key in object_dict.keys():
        if file.startswith(key):
            print(f'\n################# {key}')
            file_name = file.split('/')[-1]
            print(f'################# {file_name}\n')
            image_dir = f'image_folder/{key}_folder'
            image_name = os.listdir(image_dir)[0]
            image = Image.open(os.path.join(image_dir, image_name))
            image = image.resize((1024, 1024))
            
            load_dir = os.path.join(output_dir, file)
            pipe.load_lora_weights(load_dir)
            prompts = get_prompts(key)
            
            result_image = Image.new('RGB', (1024 * 8, 1024), color='white')
            result_image.paste(image, (0, 0))
            for i, prompt in enumerate(prompts):
                print(prompt)
                generator = torch.Generator("cuda").manual_seed(43)
                image = pipe(prompt, num_inference_steps=30, generator=generator).images[0]
                if i == 7:
                    result_image.show()
                    result_image = Image.new('RGB', (1024 * 8, 1024), color='white')
                result_image.paste(image, (((i+1)%8) * 1024, 0))

            result_image.show()
# %%
# 231225
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-48d0547b-e505-7f76-7741-a012f5c16f7a"
import torch
from safetensors.torch import load
import matplotlib.pyplot as plt
# %%
file_path = "64rank_ziplora_output/dog_and_watercolor/dog_5e-5_NoTE_1000steps_zip_5e-5_100steps/pytorch_lora_weights.safetensors"

file_path = "64rank_dreambooth_output/dog_5e-5_NoTE_1000steps/pytorch_lora_weights.safetensors"
with open(file_path, "rb") as f:
    data = f.read()

dog_state_dict = load(data)
# %%
a = dog_state_dict['unet.unet.mid_block.attentions.0.transformer_blocks.8.attn1.to_v.lora.up.weight']
b = dog_state_dict['unet.unet.mid_block.attentions.0.transformer_blocks.8.attn1.to_v.lora.down.weight']

# %%
plt.figure(figsize=(15, 8))
plt.hist(a, bins=100)
plt.show()
# %%
b = torch.hsplit(a, 10)
for x in b:
    plt.figure(figsize=(15, 8))
    plt.hist(x)
    plt.show()