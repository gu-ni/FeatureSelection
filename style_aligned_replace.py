# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-68b77852-bf59-f9d6-dad1-09a92e44c6f4"
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLPipeline_guni, DDIMScheduler
import torch
import mediapy
import sa_handler_replace
from PIL import Image
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore, AttentionReplace, LocalBlend, AttentionRefine, AttentionReweight, view_images, get_equalizer
from featurestore_utils import FeatureStoreCross
from copy import deepcopy

# init models

feature_controller = FeatureStoreCross()
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    scheduler=scheduler
).to("cuda")

# %%
#lora_kwrgs = {"feature_controller": feature_controller}
lora_kwrgs = {}
pipeline.load_lora_weights("/mnt/Face_Private-NFS2/geonhui/works2_ckpt", weight_name="dog_5e-5_NoTE_1000steps.safetensors", adapter_name="dog", **lora_kwrgs)
pipeline.load_lora_weights("/mnt/Face_Private-NFS2/geonhui/works2_ckpt", weight_name="flower_watercolor_5e-5_NoTE_1000steps.safetensors", adapter_name="style")

pipeline.set_adapters(["dog", "style"], adapter_weights=[1.0, 1.0])


# %%

handler = sa_handler_replace.Handler(pipeline)
sa_args = sa_handler_replace.StyleAlignedArgs(share_group_norm=False,
                                      share_layer_norm=False,
                                      share_attention=True,
                                      adain_queries=False,
                                      adain_keys=False,
                                      adain_values=False,
                                     )

handler.register(sa_args, None)

# %%
pipeline.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.to_q.forward_control_dict

p = [
    'a ohwx dog with blue plants and shelf on a white background in flat illustration style, minimal simple vector graphics',
    'a ohwx dog in minimal simple illustration vector graphics'
]
# %%
p = [
    'a ohwx dog, a flat illustration sticker style with white background, simple vector graphic',
    'a ohwx dog, a flat illustration sticker style with white background, simple vector graphic'
]


forward_control_dict = {
    # lora layer에서 active_adapter의 힘 조절
    "lora_control": {

        "active_adapter": ['dog'],                       # 'dog', 'style'
        "place_in_unet": ['up'],                         # 'down', 'mid', 'up'
        "self_or_cross": ['cross'],                      # 'self', 'cross'
        "attn_type": ['to_q', 'to_k', 'to_v', 'to_out'], # 'to_q', 'to_k', 'to_v', 'to_out'
        "strength": 0.0,
        "adapt_timestep": False,
        "prompt_index": [0],
    },
    # shared attention에서 attention swapping할 부분 정의
    "switch_condition": {
        "how": 'equal',
        "place_in_unet": ['up']
    }
}

same_seed = True
print('\n\n'.join(p))

if same_seed:
    print("Same Seed")
    g_cpu = torch.Generator(device='cpu')
    latents = torch.tensor([], dtype=pipeline.unet.dtype).to('cuda:0')
    for i in range(len(p)):
        g_cpu.manual_seed(0)
        temp = torch.randn(1, 4, 128, 128, device='cpu', generator=g_cpu,
                        dtype=pipeline.unet.dtype,).to('cuda:0')
        latents = torch.cat([latents, temp], dim=0)
    images = pipeline(p, latents=latents, 
                    forward_control_dict=forward_control_dict).images
    result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
    for i, image in enumerate(images):
        result_image.paste(image, (i * 1024, 0))
    result_image.show()
else:
    print("Different Seed")
    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(0)

    latents = torch.randn(len(p), 4, 128, 128, device='cpu', generator=g_cpu,
                        dtype=pipeline.unet.dtype,).to('cuda:0')
    images = pipeline(p, latents=latents, 
                    forward_control_dict=forward_control_dict).images
    result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
    for i, image in enumerate(images):
        result_image.paste(image, (i * 1024, 0))
    result_image.show()

# %% 따로따로
p = [
    'a ohwx dog, a flat illustration sticker style with white background, simple vector graphic',
    'a ohwx dog, a flat illustration sticker style'
]
# %%
p = [
    'a ohwx dog sleeping, in watercolor painting style, simple artistic painting',
    'a ohwx dog sleeping, in watercolor painting style'
]
# %%
p = [
    'a ohwx dog with blue plants and shelf on a white background in flat illustration style, minimal simple vector graphics',
    'a ohwx dog in minimal simple illustration vector graphics'
]
# %%
p = [
    'a ohwx dog riding a bicycle, a flat illustration sticker style with white background, simple vector graphic',
    'a ohwx dog riding a bicycle, a flat illustration sticker style'
]
p = [
    'a ohwx dog riding a bicycle with blue plants and shelf on a white background in flat illustration style, minimal simple vector graphics',
    'a ohwx dog riding a bicycle in minimal simple illustration vector graphics'
]
p = [
    'a ohwx dog, riding a bicycle in watercolor painting style, simple artistic painting',
    'a ohwx dog, riding a bicycle in watercolor painting style'
]
forward_control_dict = {
    "lora_control": {
        # lora layer에서 active_adapter의 힘 조절
        "forward_control_dict_one": { # style path (=첫 번째 path)

            "place_in_unet": ['up'],
            "self_or_cross": ['cross'],
            "attn_type": ['to_q', 'to_k', 'to_v', 'to_out'],
            "strength": {'dog': 0.0},
            "adapt_timestep": False,
        },
        "forward_control_dict_two": { # 나머지 path

            "place_in_unet": ['up'], 
            "self_or_cross": ['cross'], 
            "attn_type": ['to_q', 'to_k', 'to_v', 'to_out'],
            "strength": {'dog': 0.0},
            "adapt_timestep": False,
        }
    },
    # shared attention에서 attention swapping할 부분 정의
    "switch_condition": {
        "how": 'equal',
        "place_in_unet": []
    }
}

same_seed = True
print('\n\n'.join(p))

if same_seed:
    print("Same Seed")
    g_cpu = torch.Generator(device='cpu')
    latents = torch.tensor([], dtype=pipeline.unet.dtype).to('cuda:0')
    for i in range(len(p)):
        g_cpu.manual_seed(0)
        temp = torch.randn(1, 4, 128, 128, device='cpu', generator=g_cpu,
                        dtype=pipeline.unet.dtype,).to('cuda:0')
        latents = torch.cat([latents, temp], dim=0)
    images = pipeline(p, latents=latents, 
                    forward_control_dict=forward_control_dict).images
    result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
    for i, image in enumerate(images):
        result_image.paste(image, (i * 1024, 0))
    result_image.show()
else:
    print("Different Seed")
    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(0)

    latents = torch.randn(len(p), 4, 128, 128, device='cpu', generator=g_cpu,
                        dtype=pipeline.unet.dtype,).to('cuda:0')
    images = pipeline(p, latents=latents, 
                    forward_control_dict=forward_control_dict).images
    result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
    for i, image in enumerate(images):
        result_image.paste(image, (i * 1024, 0))
    result_image.show()





# %%

# %%

# %%

# %%
module_dict = dict(pipeline.unet.named_modules())

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

# %% 크기 비교로 바꾸는 용
changed_weight_dict = deepcopy(module_dict)
import torch.nn.functional as nnf
for i, key in enumerate(adapters_AB[0]):
    original_content_AB = adapters_AB[0][key]
    original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
    original_style_AB = adapters_AB[1][key]
    original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
    
    content_B_coef = original_content_AB_size / (original_content_AB_size + original_style_AB_size)  
    #content_B_coef[original_content_AB_size < original_style_AB_size] *= 2
    style_B_coef = original_style_AB_size / (original_content_AB_size + original_style_AB_size)

    back_to_content_key = key + f'.lora_B.{adapter_name[0]}'
    back_to_style_key = key + f'.lora_B.{adapter_name[1]}'

    prev_content_weight = changed_weight_dict[back_to_content_key].weight
    #masked_content_weight = torch.nn.Parameter(content_B_coef.unsqueeze(1) * prev_content_weight)
    masked_content_weight = torch.nn.Parameter(0.5 * prev_content_weight)
    changed_weight_dict[back_to_content_key].weight = masked_content_weight
    
    prev_style_weight = changed_weight_dict[back_to_style_key].weight
    #masked_style_weight = torch.nn.Parameter(style_B_coef.unsqueeze(1) * prev_style_weight)
    masked_style_weight = torch.nn.Parameter(0.5 * prev_style_weight)
    changed_weight_dict[back_to_style_key].weight = masked_style_weight

for name, module in pipeline.unet.named_modules():
    if 'lora_B.' in name:
        module.weight = changed_weight_dict[name].weight
# %%
