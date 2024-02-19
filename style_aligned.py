# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-edefb71f-8316-c832-8331-94667e51a510"
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLPipeline_guni, DDIMScheduler
import torch
from safetensors.torch import load_file
import mediapy
import sa_handler_guni
from PIL import Image
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore, AttentionReplace, LocalBlend, AttentionRefine, AttentionReweight, view_images, get_equalizer
from featurestore_utils import FeatureStoreCross
from tqdm import tqdm

# %%
# init models

#controller = AttentionStore()
#feature_controller = FeatureStoreCross()
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
#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/blue_illu_5e-5_NoTE_1000steps_revised", weight_name="pytorch_lora_weights.safetensors", adapter_name="dog")
pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/blue_illu_5e-5_NoTE_1000steps_revised_no_color", weight_name="pytorch_lora_weights.safetensors", adapter_name="style")

#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/tree_sticker_5e-5_NoTE_1000steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="style")

#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/64rank_ziplora_output_OutFeatures/dog_and_blue_illu_revised_no_color_save_merger_0.1simlam_SumAbs/dog_5e-5_NoTE_1000steps_zip_5e-3_100steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="dog", **lora_kwrgs)
#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/64rank_ziplora_output_OutFeatures/dog_and_blue_illu_revised_no_color_save_merger_0.1simlam_SumAbs/blue_illu_revised_no_color_5e-5_NoTE_1000steps_zip_5e-3_100steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="style")

#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/downloaded_lora_layer", weight_name="ParchartXL-2.0.safetensors", adapter_name="style")


# /workspace/diffusers/examples/dreambooth/64rank_ziplora_output/dog_and_blue_illu_revised_no_color_NO_MERGER/dog_5e-5_NoTE_1000steps_zip_5e-3_100steps
# %%
pipeline.set_adapters(["dog", "style"], adapter_weights=[1.0, 1.0])

# %%
pipeline.set_adapters(["style"], adapter_weights=[2.0])
# %%
percentile = 10
all_params = []
module_dict = dict(pipeline.unet.named_modules())
for key, module in tqdm(module_dict.items()):
    if 'lora_A' in key or 'lora_B' in key:
        for p in module.parameters():
            all_params.extend(p.flatten().tolist())

threshold = torch.tensor(all_params).kthvalue(int(len(all_params) * (percentile / 100))).values.item()
del all_params
threshold

for name, module in tqdm(pipeline.unet.named_modules()):
    if 'lora_A.' in name or 'lora_B.' in name:
        for p in module.parameters():
            p.data[p.data <= threshold] = 0

# %%
upper_percentile = 25
lower_percentile = 75
all_params = []
module_dict = dict(pipeline.unet.named_modules())
for key, module in tqdm(module_dict.items()):
    if 'lora_A' in key or 'lora_B' in key:
        for p in module.parameters():
            all_params.extend(p.flatten().tolist())

upper_threshold = torch.tensor(all_params).kthvalue(int(len(all_params) * (1 - upper_percentile / 100))).values.item()
lower_threshold = torch.tensor(all_params).kthvalue(int(len(all_params) * (1 - lower_percentile / 100))).values.item()

del all_params
print(upper_threshold)
print(lower_threshold)
print(upper_threshold == -lower_threshold)

for name, module in tqdm(pipeline.unet.named_modules()):
    if 'lora_A.' in name or 'lora_B.' in name:
        for p in module.parameters():
            p.data[(p.data < upper_threshold) & (p.data > lower_threshold)] = 0

# %%
for name, module in tqdm(pipeline.unet.named_modules()):
    if 'lora_A.' in name or 'lora_B.' in name:
    #if 'lora_B.' in name:
        for p in module.parameters():
            """
            numel = p.numel()
            num_elements_to_replace = int(0.5 * numel)
            indices_to_replace = torch.randperm(numel)[:num_elements_to_replace]

            row_indices = indices_to_replace // p.size(1)
            col_indices = indices_to_replace % p.size(1)
            p.data[row_indices, col_indices] = 0
            """
            p.data += torch.randn_like(p.data) * 0.01
# %%
#original_dict = dict(pipeline.unet.named_modules())
module_dict = dict(pipeline.unet.named_modules())
# %%
lora_name = ['lora_A', 'lora_B']
adapter_name = ['dog', 'style']
n = 0

adapters_A_and_B = [{}, {}]
attn_heads = {}
for key, module in module_dict.items():
    if 'lora_A' in key or 'lora_B' in key:
        if adapter_name[0] in key:
            for p in module.parameters():
                adapters_A_and_B[0][key] = p
        elif adapter_name[1] in key:
            for p in module.parameters():
                adapters_A_and_B[1][key] = p
    if hasattr(module, 'heads') and module.heads != None:
        attn_heads[key] = module.heads

adapters_AB = [{}, {}]
for i, dic in enumerate(adapters_A_and_B):
    for name, module in dic.items():
        if 'lora_A' in name:
            name_A = name
            module_A = module
        elif ('lora_B' in name) and (name.split('.')[:-2] == name_A.split('.')[:-2]):
            module_AB = torch.mm(module_A.T, module.T)
            adapters_AB[i]['.'.join(name.split('.')[:-2])] = module_AB

# %%
# dynamic_ratio 이용해서 바꾸는 용
dynamic_ratio = 0.25
import torch.nn.functional as nnf
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
    
    vec_len = cos_sim.shape[0]
    threshold = int(vec_len * dynamic_ratio)
    sorted_idx = torch.argsort(cos_sim)
    top_idx = sorted_idx[-threshold:]
    bottom_idx = sorted_idx[:threshold]
    switch_mask = torch.zeros_like(cos_sim)
    switch_mask[top_idx] = 1
    switch_mask[bottom_idx] = 1
    
    switch_mask = torch.ones_like(cos_sim)
    
    content_B_coef = torch.ones_like(content_bigger_than_style)
    content_B_coef[(switch_mask == 1) & (content_bigger_than_style == 0)] = 0
    
    # 추가
    #content_B_coef[] = 
    
    style_B_coef = torch.ones_like(content_bigger_than_style)
    style_B_coef[(switch_mask == 1) & (content_bigger_than_style == 1)] = 0
    

    back_to_content_key = key + f'.lora_B.{adapter_name[0]}'
    back_to_style_key = key + f'.lora_B.{adapter_name[1]}'

    prev_content_weight = module_dict[back_to_content_key].weight
    masked_content_weight = torch.nn.Parameter(content_B_coef.unsqueeze(1) * prev_content_weight)
    #masked_content_weight = torch.nn.Parameter(0.55 * prev_content_weight)
    module_dict[back_to_content_key].weight = masked_content_weight
    
    prev_style_weight = module_dict[back_to_style_key].weight
    masked_style_weight = torch.nn.Parameter(style_B_coef.unsqueeze(1) * prev_style_weight)
    #masked_style_weight = torch.nn.Parameter(0.55 * prev_style_weight)
    module_dict[back_to_style_key].weight = masked_style_weight

for name, module in pipeline.unet.named_modules():
    if 'lora_B.' in name:
        module.weight = module_dict[name].weight

# %% 크기 비교로 바꾸는 용
import torch.nn.functional as nnf
for i, key in enumerate(adapters_AB[0]):
    original_content_AB = adapters_AB[0][key]
    original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
    original_style_AB = adapters_AB[1][key]
    original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
    
    content_B_coef = original_content_AB_size / (original_content_AB_size + original_style_AB_size)  
    content_B_coef[original_content_AB_size < original_style_AB_size] *= 2
    style_B_coef = 2 * original_style_AB_size / (original_content_AB_size + original_style_AB_size)

    back_to_content_key = key + f'.lora_B.{adapter_name[0]}'
    back_to_style_key = key + f'.lora_B.{adapter_name[1]}'

    prev_content_weight = module_dict[back_to_content_key].weight
    masked_content_weight = torch.nn.Parameter(content_B_coef.unsqueeze(1) * prev_content_weight)
    #masked_content_weight = torch.nn.Parameter(0.55 * prev_content_weight)
    module_dict[back_to_content_key].weight = masked_content_weight
    
    prev_style_weight = module_dict[back_to_style_key].weight
    masked_style_weight = torch.nn.Parameter(style_B_coef.unsqueeze(1) * prev_style_weight)
    #masked_style_weight = torch.nn.Parameter(0.55 * prev_style_weight)
    module_dict[back_to_style_key].weight = masked_style_weight

for name, module in pipeline.unet.named_modules():
    if 'lora_B.' in name:
        module.weight = module_dict[name].weight

# %%
# (기존) 그래프 찍어보는 용
import torch.nn.functional as nnf

for at in ['to_q', 'to_k', 'to_v', 'to_out']:
    n = 0
    print(at)
    fig, axes = plt.subplots(1, 4, figsize=(25, 4))
    for i, key in enumerate(adapters_AB[0]):
        if 'up_blocks.0.attentions.0' in key and 'attn2' in key and at in key:
            if 'transformer_blocks.0' in key or 'transformer_blocks.3' in key or 'transformer_blocks.6' in key or 'transformer_blocks.9' in key:
                
                original_content_AB = adapters_AB[0][key]
                original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
                original_style_AB = adapters_AB[1][key]
                original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
                
                content = nnf.normalize(original_content_AB, dim=0)
                style = nnf.normalize(original_style_AB, dim=0)
                cos_sim = (content * style).sum(dim=0)
                
                axes[n].set_title(f'{key}')
                axes[n].hist(cos_sim.detach().cpu().numpy(), bins=100)
                axes[n].set_xlim([-1.0, 1.0])
                
                n += 1
                """
                plt.figure(figsize=(15, 8))
                plt.title(f'{key}')
                plt.hist(cos_sim.detach().cpu().numpy(), bins=100)
                plt.xlim([-1.0, 1.0])
                plt.show()
                """
                if n == 10:
                    print('BREAK')
                    break
    plt.tight_layout()
    plt.show()
# %%

# %%
# 그래프 찍어보는 용
import torch.nn.functional as nnf
n = 0
for i, key in enumerate(adapters_AB[1]):
    #if 'down_blocks.1.attentions.0.transformer_blocks' in key and 'attn2' in key and 'to_out' in key:
    if key == 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_v':
        n += 1
        original_content_AB = adapters_AB[0][key]
        original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
        original_style_AB = adapters_AB[1][key]
        original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
        
        content_bigger_than_style = torch.zeros_like(original_content_AB_size)
        content_bigger_than_style[original_content_AB_size > original_style_AB_size] = 1
        
        content = nnf.normalize(original_content_AB, dim=0)
        style = nnf.normalize(original_style_AB, dim=0)
        cos_sim = (content * style).sum(dim=0)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        axes[0].set_title(f'{key}')
        axes[0].hist(cos_sim.detach().cpu().numpy(), bins=100)
        axes[0].set_xlim([-1.0, 1.0])
        
        c_size = original_content_AB_size.detach().cpu().numpy()
        s_size = original_style_AB_size.detach().cpu().numpy()
        axes[1].hist(c_size, bins=100, color='orange', alpha=0.7, label='content')
        axes[1].hist(s_size, bins=100, color='green', alpha=0.7, label='style')
        axes[1].set_xlim([0.0, 0.4])
        axes[1].legend(fontsize=15)
        axes[1].set_title(f'{c_size.mean()} | {s_size.mean()}')
        plt.tight_layout()
        plt.show()
        
        if n == 10:
            print('BREAK')
            break

# %%
# ranking 매기는 용
c_max_size = 0.0
s_max_size = 0.0
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
    
    c_size_mean = original_content_AB_size.detach().cpu().numpy().mean()
    s_size_mean = original_style_AB_size.detach().cpu().numpy().mean()

    if c_size_mean > c_max_size:
        c_max_size = c_size_mean
        c_max_key = key
    
    if s_size_mean > s_max_size:
        s_max_size = s_size_mean
        s_max_key = key

# %%
original_list = [9, 4, 2, 7, 1, 10, 8, 5, 3, 6]

top_5_list = []

for num in original_list:
    if len(top_5_list) < 5:
        top_5_list.append(num)
    else:
        minimum = min(top_5_list)
        if num > minimum:
            top_5_list.remove(minimum)
            top_5_list.append(num)

# 결과 출력
print("원본 리스트:", original_list)
print("상위 5개의 정수 리스트:", sorted(top_5_list))  # 정렬하여 출력

# %%
# 상위 n개 추리는 용 (AB 행렬 기준)
import numpy as np
top_count = 16
top_list = []
for i, key in enumerate(adapters_AB[1]):
    AB_tensor = adapters_AB[1][key]
    AB_col_size = torch.norm(AB_tensor, p=2, dim=0)
    AB_col_size_max = AB_col_size.max().detach().cpu().item()
    
    if len(top_list) < top_count:
        top_list.append((key, AB_col_size_max))
    else:
        minimum = min(list(zip(*top_list))[1])
        argmin = np.argmin(list(zip(*top_list))[1])
        if AB_col_size_max > minimum:
            del top_list[argmin]
            top_list.append((key, AB_col_size_max))
top_list.sort(key=lambda x: x[1], reverse=True)
#list(zip(*top_list))[1]
top_list
# %%
# 상위 n개 추리는 용 (column 기준)
import numpy as np
top_count = 2048
top_list = []
for i, key in tqdm(enumerate(adapters_AB[1])):
    AB_tensor = adapters_AB[1][key]
    AB_col_size = torch.norm(AB_tensor, p=2, dim=0)
    top_indices = torch.argsort(AB_col_size, descending=True)[:top_count]
    top_elements = AB_col_size[top_indices].detach().cpu().numpy()
    top_indices = top_indices.detach().cpu().numpy()
    
    for j, col_size in zip(top_indices, top_elements):
        #j = j.detach().cpu().item()
        #col_size = col_size.item()
        if len(top_list) < top_count:
            top_list.append((key, j, col_size))
        else:
            argmin = np.argmin(list(zip(*top_list))[-1])
            minimum = top_list[argmin][-1]
            if top_elements.max() < minimum:
                break
            if col_size > top_list[argmin][-1]:
                del top_list[argmin]
                top_list.append((key, j, col_size))
top_list.sort(key=lambda x: x[-1], reverse=True)
top_list

# %%
from collections import Counter
c = Counter(list(zip(*top_list))[0]).most_common()

top_dict = {layer: [] for layer in list(zip(*c))[0]}

for col in top_list:
    top_dict[col[0]].append(col[1])
# %%
# column별 크기 정렬 확인 후 weight 크기 plot 찍어보는 용
import torch.nn.functional as nnf
for i, key in enumerate(adapters_AB[1]):
    if key == 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v':
        AB_tensor = adapters_AB[1][key]
        AB_col_size = torch.norm(AB_tensor, p=2, dim=0)
        
        plt.figure(figsize=(15, 8))
        for i in range(0, AB_col_size.shape[0]+1, 64):
            plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
        
        plt.plot(AB_col_size.detach().cpu().numpy())
        plt.show()
        
# %% colimn 크기 기준 zero vector로 교체
original_dict = dict(pipeline.unet.named_modules())

# %%
target_key = list(zip(*c))[0]

for i, key in tqdm(enumerate(adapters_AB[1])):
    for layer in top_dict:
        if key == layer:
            target_indices = top_dict[layer]
            AB_tensor = adapters_AB[1][key]
            AB_col_size = torch.norm(AB_tensor, p=2, dim=0)
            
            B_coef = torch.ones_like(AB_col_size)
            B_coef[target_indices] = 0

            back_to_key = key + f'.lora_B.{adapter_name[1]}'

            prev_style_weight = module_dict[back_to_key].weight
            masked_style_weight = torch.nn.Parameter(B_coef.unsqueeze(1) * prev_style_weight)
            module_dict[back_to_key].weight = masked_style_weight

for name, module in pipeline.unet.named_modules():
    if 'lora_B.' in name:
        module.weight = module_dict[name].weight
        
        
# %%
div = torch.divide(original_style_AB, original_content_AB)
mean = div.mean(dim=0).detach().cpu().numpy()
std = div.std(dim=0).detach().cpu().numpy()

plt.figure(figsize=(15, 8))
plt.plot(mean)
plt.show()

plt.figure(figsize=(15, 8))
plt.plot(std)
plt.show()
# %%

for name, module in pipeline.unet.named_modules():
    if 'lora_B.' in name:
        module.weight = module_dict[name].weight

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

style_description = ', with blue plants and shelf on a white background in flat cartoon illustration style, minimal simple vector graphics'
style_description = ''
style_prefix = f'a dog'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'

set_p = set_of_prompts(style_description, style_prefix)
best_seed = 0
set_p
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

style_description = 'ral-3dwvz, '
style_prefix = f'statue'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'


set_p = set_of_prompts(style_description, style_prefix)
best_seed = 0
set_p

# %% content 없이 style만
def set_of_prompts(style_description, style_prefix):
    style_prompt = f'{style_prefix}{style_description}'
    sets_of_prompts1 = [
        style_prompt,
        f"a dog{style_description}",
        f"an orange{style_description}",
        f"clouds{style_description}",
    ]
    
    sets_of_prompts2 = [
        f"a man{style_description}",
        f"a baby penguin{style_description}",
        f"a moose{style_description}",
        f"a towel{style_description}"
    ]

    sets_of_prompts3 = [
        f"an espresso machine{style_description}",
        f"an avocado{style_description}",
        f"a crown{style_description}",
        f"the Golden Gate bridge{style_description}",
    ]
    return [sets_of_prompts1, sets_of_prompts2, sets_of_prompts3]

style_description = ' with blue plants and shelf on a white background in flat cartoon illustration style, minimal simple vector graphics'
style_prefix = f'a woman'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'


set_p = set_of_prompts(style_description, style_prefix)
best_seed = 0
set_p
# %%
def set_of_prompts(style_description, style_prefix):
    style_prompt = f'{style_description}{style_prefix}'
    sets_of_prompts1 = [
        style_prompt,
        f"{style_description}a dog",
        f"{style_description}an orange",
        f"{style_description}clouds",
    ]
    
    sets_of_prompts2 = [
        f"{style_description}a man",
        f"{style_description}a baby penguin",
        f"{style_description}a moose",
        f"{style_description}a towel"
    ]

    sets_of_prompts3 = [
        f"{style_description}an espresso machine",
        f"{style_description}an avocado",
        f"{style_description}a crown",
        f"{style_description}the Golden Gate bridge",
    ]
    return [sets_of_prompts1, sets_of_prompts2, sets_of_prompts3]

style_description = 'on parchment '
style_prefix = f'a dragon'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'


set_p = set_of_prompts(style_description, style_prefix)
best_seed = 0




# %%
def set_of_prompts(content_description, style_description, style_prefix):
    style_prompt = f'{style_prefix}{style_description}'
    sets_of_prompts1 = [
        style_prompt,
        f"{content_description}{style_description}",
        f"{content_description} playing with a ball{style_description}",
        f"{content_description} catching a frisbie{style_description}",
    ]
    
    sets_of_prompts2 = [
        f"{content_description} wearing a hat{style_description}",
        f"{content_description} with a crown{style_description}",
        f"{content_description} riding a bicycle{style_description}",
    ]

    sets_of_prompts3 = [
        f"{content_description} sleeping{style_description}",
        f"{content_description} in a boat{style_description}",
        f"{content_description} driving a car{style_description}",
    ]
    return [sets_of_prompts1, sets_of_prompts2, sets_of_prompts3]

content_description = 'a ohwx dog'
style_description = ', with blue plants and shelf on a white background in flat cartoon illustration style, minimal simple vector graphics'
style_prefix = f'a woman'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'

set_p = set_of_prompts(content_description, style_description, style_prefix)
best_seed = 0
set_p

# %% DIFFERENT SEED
seeds_list = [[best_seed, 10, 20, 30], [40, 50, 60], [70, 80, 90]]
all_images = Image.new('RGB', (1024 * 10, 1024), color='white')

# %%
seeds_list = [[best_seed, 10, 20, 30], [40, 50, 60, 70], [80, 90, 100, 110]]
all_images = Image.new('RGB', (1024 * 12, 1024), color='white')
# %%
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

del temp
del latents
del image
del result_image
del all_images
# %%
content_prompt = ['a dog',
                  'an orange',
                  'a baby penguin',
                  'an avocado',
                  'a crown']
prompts = [f'{content} with blue plants and shelf on a white background in flat cartoon illustration style, minimal simple vector graphics' for content in content_prompt]

woman_seed = [0, 1, 2, 3, 4]
all_images = Image.new('RGB', (1024 * len(woman_seed), 1024), color='white')


for i, seed in enumerate(woman_seed):
    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(seed)
    print(prompts[i])
    image = pipeline(prompts[i], generator=g_cpu).images[0]
    image.show()
    all_images.paste(image, (i * 1024, 0))
all_images.show()

# %%
from safetensors.torch import load_file
import matplotlib.pyplot as plt

mergers = []
content_merger_path = "/workspace/diffusers/examples/dreambooth/64rank_ziplora_output_OutFeatures/dog_and_blue_illu_revised_no_color_save_merger_0.1simlam_SumAbs/dog_merger/merger.safetensors"
style_merger_path = "/workspace/diffusers/examples/dreambooth/64rank_ziplora_output_OutFeatures/dog_and_blue_illu_revised_no_color_save_merger_0.1simlam_SumAbs/blue_illu_revised_no_color_merger/merger.safetensors"
#content_no_merger_path = "/workspace/diffusers/examples/dreambooth/64rank_ziplora_output/dog_and_blue_illu_revised_no_color_NO_MERGER_save_merger/dog_merger/merger.safetensors"


content_merger = load_file(content_merger_path)
style_merger = load_file(style_merger_path)
#content_no_merger = load_file(content_no_merger_path)

# %%
plt.figure(figsize=(15, 8))

name = 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_out.0'
plt.plot(content_merger[f'{name}.merger_1'].detach().cpu().numpy(), label='content')
plt.plot(style_merger[f'{name}.merger_2'].detach().cpu().numpy(), label='style')
plt.title(name, fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.show()
# %%
key_list = []
for key in content_merger:
    if 'up_blocks.0.attentions.0.transformer_blocks' in key and 'attn2' in key and 'to_k' in key:
        #print(key)
        key_list.append(key[:-9])


plt.figure(figsize=(20, 10))
for i in range(0, content_merger[f'{key_list[0]}.merger_1'].shape[0]+1, 64):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

for i, k in enumerate(key_list):
    c_color = plt.cm.Reds((i+1) / (len(key_list)))
    s_color = plt.cm.Blues((i+1) / (len(key_list)))
    plt.plot(content_merger[f'{k}.merger_1'].detach().cpu().numpy(), 
             color=c_color, 
             alpha=0.7,
             label='content' if i == len(key_list)-1 else None)
    plt.plot(style_merger[f'{k}.merger_2'].detach().cpu().numpy(), 
             color=s_color, 
             alpha=0.7, 
             label='style' if i == len(key_list)-1 else None)
#plt.title(k, fontsize=20)
plt.title(k, fontsize=20)
plt.legend()
#plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.show()
# %%
import matplotlib.ticker as ticker

key_list = []
for key in content_merger:
    if 'up_blocks.0.attentions.2.transformer_blocks' in key and 'attn2' in key and 'to_q' in key:
        #print(key)
        key_list.append(key[:-9])


plt.figure(figsize=(20, 10))
for i in range(0, content_merger[f'{key_list[0]}.merger_1'].shape[0]+1, 64):
    plt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

for i, k in enumerate(key_list):
    c_color = plt.cm.viridis((i) / (len(key_list)-1))
    plt.plot(content_merger[f'{k}.merger_1'].detach().cpu().numpy(), 
             color=c_color, 
             alpha=0.7,
             label='content' if i == len(key_list)-1 else None)
""
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)  # 지수 표기 비활성화
plt.gca().yaxis.set_major_formatter(formatter)
plt.ticklabel_format(style='plain', axis='y')
#plt.title(k, fontsize=20)""
plt.title(k, fontsize=20)
plt.legend()
#plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.show()
# %%
content_merger['up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_k.merger_1'][0].item()
# %%
