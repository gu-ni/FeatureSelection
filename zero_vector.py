# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-5fd619b1-1513-c518-8152-a8c4e6ea3ce5"
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLPipeline_guni, DDIMScheduler
import torch
import torch.nn.functional as nnf
import sa_handler_guni
from PIL import Image
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore, AttentionReplace, LocalBlend, AttentionRefine, AttentionReweight, view_images, get_equalizer
from featurestore_utils import FeatureStoreCross
import seaborn as sns

# %%
# init models

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
pipeline = StableDiffusionXLPipeline_guni.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    scheduler=scheduler
).to("cuda")

# %%
#lora_kwrgs = {"feature_controller": feature_controller}
lora_kwrgs = {}
pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/dog_5e-5_NoTE_1000steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="dog", **lora_kwrgs)
#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/blue_illu_5e-5_NoTE_1000steps_long_prompt", weight_name="pytorch_lora_weights.safetensors", adapter_name="style")
pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/tree_sticker_5e-5_NoTE_1000steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="style")
#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/64rank_ziplora_output/dog_and_tree_sticker/dog_5e-5_NoTE_1000steps_zip_5e-3_100steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="dog", **lora_kwrgs)
#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/64rank_ziplora_output/dog_and_tree_sticker/tree_sticker_5e-5_NoTE_1000steps_zip_5e-3_100steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="style")

pipeline.set_adapters(["dog", "style"], adapter_weights=[1.0, 1.0])

# %%
module_dict = dict(pipeline.unet.named_modules())

lora_name = ['lora_A', 'lora_B']
adapter_name = ['dog', 'style']

adapters_A_and_B = [{}, {}]
for key, module in module_dict.items():
    if 'lora_A' in key or 'lora_B' in key:
        if adapter_name[0] in key:
            for p in module.parameters():
                adapters_A_and_B[0][key] = p
        elif adapter_name[1] in key:
            for p in module.parameters():
                adapters_A_and_B[1][key] = p

adapters_AB = [{}, {}]
for i, dic in enumerate(adapters_A_and_B):
    for name, module in dic.items():
        if 'lora_A' in name:
            name_A = name
            module_A = module
        elif ('lora_B' in name) and (name.split('.')[:-2] == name_A.split('.')[:-2]):
            module_AB = torch.mm(module_A.T, module.T)
            adapters_AB[i]['.'.join(name.split('.')[:-2])] = module_AB

# %% # 그래프 찍어보는 용 - weight 자체 찍어봄
ratio = 0.01
dynamic_ratio = 0.6
n = 0
for i, key in enumerate(adapters_AB[0]):
    if 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k' in key:
        n += 1
        if n >= 0:
            original_content_AB = adapters_AB[0][key]
            original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
            original_style_AB = adapters_AB[1][key]
            original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
            
            content_bigger_than_style = torch.zeros_like(original_content_AB_size)
            content_bigger_than_style[original_content_AB_size > original_style_AB_size] = 1
            
            content = nnf.normalize(original_content_AB, dim=0)
            style = nnf.normalize(original_style_AB, dim=0)
            cos_sim = (content * style).sum(dim=0)

            # dynamic threshold
            vec_len = cos_sim.shape[0]
            threshold = int(vec_len * dynamic_ratio)
            sorted_idx = torch.argsort(cos_sim)
            top_idx = sorted_idx[-threshold:]
            bottom_idx = sorted_idx[:threshold]
            switch_mask = torch.zeros_like(cos_sim)
            switch_mask[top_idx] = 1
            switch_mask[bottom_idx] = 1
            
            content_B_coef = torch.ones_like(content_bigger_than_style)
            content_B_coef[(switch_mask == 1) & (content_bigger_than_style == 0)] = 0
            
            style_B_coef = torch.ones_like(content_bigger_than_style)
            style_B_coef[(switch_mask == 1) & (content_bigger_than_style == 1)] = 0
            
            print(torch.count_nonzero(switch_mask))
            print(switch_mask.shape[0] - torch.count_nonzero(switch_mask))
            print(key)
            print(cos_sim.shape)
            plt.figure(figsize=(10, 6))
            plt.hist(cos_sim.detach().cpu().numpy(), bins=100)
            plt.show()

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for j in range(4):
                sub_name = f'{lora_name[j % 2]}.{adapter_name[j // 2]}'
                tensor_name = f'{key}.{sub_name}'
                tensor = adapters_A_and_B[j // 2][tensor_name].T.detach().cpu().numpy()
                sns.heatmap(tensor, ax=axes[j])
                axes[j].set_title(f'{sub_name} {tensor.shape}')
            plt.suptitle(key, fontsize=16)
            plt.tight_layout()
            plt.show()
            
            
            fig, axes = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[10, 10 ,1]), figsize=(15, 20))
            
            vmin = min(adapters_AB[0][key].detach().cpu().numpy().min(),
                       adapters_AB[1][key].detach().cpu().numpy().min())
            vmax = max(adapters_AB[0][key].detach().cpu().numpy().max(),
                       adapters_AB[1][key].detach().cpu().numpy().max())
             
            
            for k in range(2):
                tensor = adapters_AB[k][key].detach().cpu().numpy()
                sns.heatmap(tensor, ax=axes[k], cbar=False, vmin=vmin, vmax=vmax)
                axes[k].set_title(f'{adapter_name[k]} {tensor.shape}')
            plt.suptitle(key, fontsize=16)
            fig.colorbar(axes[1].collections[0], cax=axes[2])
            plt.tight_layout()
            plt.show()
            
            if n == 20:
                print('BREAK!')
                break
        break
# %%
# %% # 그래프 찍어보는 용 - column 크기 비교
import numpy as np
ratio = 0.01
dynamic_ratio = 0.6
n = 0
for i, key in enumerate(adapters_AB[0]):
    if 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v' in key:
        n += 1
        if n >= 0:
            original_content_AB = adapters_AB[0][key]
            original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
            original_style_AB = adapters_AB[1][key]
            original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
            
            content_bigger_than_style = torch.zeros_like(original_content_AB_size)
            content_bigger_than_style[original_content_AB_size > original_style_AB_size] = 1
            
            content = nnf.normalize(original_content_AB, dim=0)
            style = nnf.normalize(original_style_AB, dim=0)
            cos_sim = (content * style).sum(dim=0)

            # dynamic threshold
            vec_len = cos_sim.shape[0]
            threshold = int(vec_len * dynamic_ratio)
            sorted_idx = torch.argsort(cos_sim)
            top_idx = sorted_idx[-threshold:]
            bottom_idx = sorted_idx[:threshold]
            switch_mask = torch.zeros_like(cos_sim)
            switch_mask[top_idx] = 1
            switch_mask[bottom_idx] = 1
            
            content_B_coef = torch.ones_like(content_bigger_than_style)
            content_B_coef[(switch_mask == 1) & (content_bigger_than_style == 0)] = 0
            
            style_B_coef = torch.ones_like(content_bigger_than_style)
            style_B_coef[(switch_mask == 1) & (content_bigger_than_style == 1)] = 0
            cos_sim = cos_sim.detach().cpu().numpy()
            
            print(torch.count_nonzero(switch_mask))
            print(switch_mask.shape[0] - torch.count_nonzero(switch_mask))
            print(key)
            print(cos_sim.shape)
            
            # cosine similarity histogram
            plt.figure(figsize=(10, 6))
            plt.hist(cos_sim, bins=100)
            #plt.bar(np.arange(640), cos_sim)
            plt.title('Histogram of Cosine Similarity between Content & Style')
            plt.show()
            
            # sorted cosine similarity
            plt.figure(figsize=(10, 6))
            cos_sim_sorted_idx = np.argsort(cos_sim)
            sorted_y = [cos_sim[idx] for idx in cos_sim_sorted_idx]
            plt.bar(np.arange(len(cos_sim)), sorted_y, width=1)
            plt.title('Sorted Cosine Similarity between Content & Style')
            plt.show()
            
            
            original_content_AB_size = original_content_AB_size.detach().cpu().numpy()
            original_style_AB_size = original_style_AB_size.detach().cpu().numpy()

            threshold = np.percentile(original_content_AB_size, 80)
            upper_20_percent = original_content_AB_size[original_content_AB_size >= threshold]
            upper_idx = np.where(original_content_AB_size >= threshold)[0]
            """
            plt.figure(figsize=(10, 6))
            #plt.hist(original_content_AB_size, bins=200)
            #plt.hist(upper_20_percent, bins=200, color='orange')
            plt.bar(np.arange(640), original_content_AB_size)
            #plt.bar(upper_idx, upper_20_percent, color='orange')
            plt.show()
            """
            
            # histogram of column size
            fig, axes = plt.subplots(2, 1, figsize=(10, 10))
            for i, col_size in enumerate([original_content_AB_size, original_style_AB_size]):
                plt.hist(col_size, bins=100, ax=axes[i])
                axes[i].sub_titles(f'{adapter_name[i]}')
            plt.title('Histogram of Content Column Size')
            plt.show()
            
            
            
            # sorted column size
            plt.figure(figsize=(10, 6))
            size_sorted_idx = np.argsort(original_content_AB_size)
            sorted_y = [original_content_AB_size[idx] for idx in size_sorted_idx]
            plt.bar(np.arange(len(original_content_AB_size)), sorted_y, width=1)
            
            size_upper_sorted_idx = np.argsort(upper_20_percent)
            sorted_y = [upper_20_percent[idx] for idx in size_upper_sorted_idx]
            plt.bar(np.arange(len(original_content_AB_size) - len(upper_20_percent), len(original_content_AB_size)), sorted_y, width=1)
            plt.title('Sorted Column Size')
            plt.show()
            
            
            if n == 2:
                print('BREAK!')
                break
        break

# %%

# %% weight 바꾸는 용
ratio = 0.01
dynamic_ratio = 0.3
n = 0
for i, key in enumerate(adapters_AB[0]):
    original_content_AB = adapters_AB[0][key]
    original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
    original_style_AB = adapters_AB[1][key]
    original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
    
    content_bigger_than_style = torch.zeros_like(original_content_AB_size)
    content_bigger_than_style[original_content_AB_size > original_style_AB_size] = 1
    
    content = nnf.normalize(original_content_AB, dim=0)
    style = nnf.normalize(original_style_AB, dim=0)
    cos_sim = (content * style).sum(dim=0)
    """
    # static threshold
    switch_mask = torch.zeros_like(cos_sim)
    switch_mask[(cos_sim >= ratio) | (cos_sim <= -ratio)] = 1
    """
    # dynamic threshold
    vec_len = cos_sim.shape[0]
    threshold = int(vec_len * dynamic_ratio)
    sorted_idx = torch.argsort(cos_sim)
    top_idx = sorted_idx[-threshold:]
    bottom_idx = sorted_idx[:threshold]
    switch_mask = torch.zeros_like(cos_sim)
    switch_mask[top_idx] = 1
    switch_mask[bottom_idx] = 1
    
    content_B_coef = torch.ones_like(content_bigger_than_style)
    content_B_coef[(switch_mask == 1) & (content_bigger_than_style == 0)] = 0
    #content_B_coef[switch_mask == 1] = 0
    
    style_B_coef = torch.ones_like(content_bigger_than_style)
    style_B_coef[(switch_mask == 1) & (content_bigger_than_style == 1)] = 0
    
    back_to_content_key = key + f'.lora_B.{adapter_name[0]}'
    back_to_style_key = key + f'.lora_B.{adapter_name[1]}'

    prev_content_weight = module_dict[back_to_content_key].weight
    masked_content_weight = torch.nn.Parameter(content_B_coef.unsqueeze(1) * prev_content_weight)
    module_dict[back_to_content_key].weight = masked_content_weight
    
    prev_style_weight = module_dict[back_to_style_key].weight
    masked_style_weight = torch.nn.Parameter(style_B_coef.unsqueeze(1) * prev_style_weight)
    module_dict[back_to_style_key].weight = masked_style_weight
    

# %%

for name, module in pipeline.unet.named_modules():
    if 'lora_B.' in name:
        module.weight = module_dict[name].weight

del module_dict
del adapters_A_and_B
del adapters_AB


# %% PROMPT SETTING
def set_of_prompts(style_description, style_prefix):
    style_prompt = f'{style_prefix}{style_description}'
    sets_of_prompts1 = [
        style_prompt,
        f"a ohwx dog{style_description}",
        f"a ohwx dog playing with a ball{style_description}",
        f"a ohwx dog catching a frisbie{style_description}",
    ]
    
    sets_of_prompts2 = [
        f"a ohwx dog wearing a hat{style_description}",
        f"a ohwx dog with a crown{style_description}",
        f"a ohwx dog riding a bicycle{style_description}",
    ]

    sets_of_prompts3 = [
        f"a ohwx dog sleeping{style_description}",
        f"a ohwx dog in a boat{style_description}",
        f"a ohwx dog driving a car{style_description}",
    ]
    return [sets_of_prompts1, sets_of_prompts2, sets_of_prompts3]

style_description = ' in minimal simple illustration, vector graphics'
style_prefix = f'a woman'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'


set_p = set_of_prompts(style_description, style_prefix)
best_seed = 0
# %%
def set_of_prompts(style_description, style_prefix):
    style_prompt = f'{style_description}{style_prefix}'
    sets_of_prompts1 = [
        style_prompt,
        f"{style_description}a ohwx dog",
        f"{style_description}a ohwx dog playing with a ball",
        f"{style_description}a ohwx dog catching a frisbie",
    ]
    
    sets_of_prompts2 = [
        f"{style_description}a ohwx dog wearing a hat",
        f"{style_description}a ohwx dog with a crown",
        f"{style_description}a ohwx dog riding a bicycle",
    ]

    sets_of_prompts3 = [
        f"{style_description}a ohwx dog sleeping",
        f"{style_description}a ohwx dog in a boat",
        f"{style_description}a ohwx dog driving a car",
    ]
    return [sets_of_prompts1, sets_of_prompts2, sets_of_prompts3]

style_description = 'a flat illustration style sticker of '
style_prefix = f'Christmas tree'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'


set_p = set_of_prompts(style_description, style_prefix)
best_seed = 0

# %% DIFFERENT SEED
seeds_list = [[best_seed, 10, 20, 30], [40, 50, 60], [70, 80, 90]]
all_images = Image.new('RGB', (1024 * 10, 1024), color='white')
n = 0
for k in range(len(set_p)):
    p = set_p[k]
    print('\n'.join(p))
    latents = torch.tensor([], dtype=pipeline.unet.dtype,)
    for seed in seeds_list[k]:
        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(seed)
        temp = torch.randn(1, 4, 128, 128, device='cpu', generator=g_cpu,
                                dtype=pipeline.unet.dtype,)
        latents = torch.cat([latents, temp], dim=0)

    images = pipeline(p, latents=latents).images

    result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
    for i, image in enumerate(images):
        result_image.paste(image, (i * 1024, 0))
        all_images.paste(image, (n * 1024, 0))
        n += 1
    result_image.show()
all_images.show()
# %%\
    
a = torch.ones(8)
b = torch.ones(8)
a += torch.randn(8, generator=torch.Generator().manual_seed(0))
b += torch.randn(8, generator=torch.Generator().manual_seed(1))
b = b - b.dot(a) * a / a.dot(a)
#a = a / torch.norm(a)
#b = b / torch.norm(b)

A = nnf.normalize(torch.randn([5, 8]), dim=0)
B = nnf.normalize(torch.randn([5, 8]), dim=0)

# %%
print(f'{a}\n{b}\n{A}\n{B}')
print(f'a: {a.shape}\nb: {b.shape}\nA: {A.shape}\nB: {B.shape}')
# %%
(A * B).sum(0)
# %%
((A * a) * (B * b)).sum(0)
# %%
import torch