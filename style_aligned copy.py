# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-7e5f8113-893b-23c8-cd67-101eea2f8eea"
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLPipeline_guni, DDIMScheduler
import torch
import mediapy
import sa_handler_guni
from PIL import Image
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore, AttentionReplace, LocalBlend, AttentionRefine, AttentionReweight, view_images, get_equalizer
from featurestore_utils import FeatureStoreCross

# init models

controller = AttentionStore()
feature_controller = FeatureStoreCross()
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
#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/blue_illu_5e-5_NoTE_1000steps_long_prompt", weight_name="pytorch_lora_weights.safetensors", adapter_name="blue_illu")
pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/output_dir_64rank/NoTE/blue_illu_5e-5_NoTE_1000steps_revised_no_color", weight_name="pytorch_lora_weights.safetensors", adapter_name="style")

#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/64rank_ziplora_output/dog_and_blue_illu/dog_5e-5_NoTE_1000steps_zip_1e-3_100steps", weight_name="pytorch_lora_weights.safetensors", adapter_name="dog")
#pipeline.load_lora_weights("/workspace/diffusers/examples/dreambooth/64rank_ziplora_output/dog_and_blue_illu/blue_illu_5e-5_NoTE_1000steps_zip_1e-3_100steps_long_prompt", weight_name="pytorch_lora_weights.safetensors", adapter_name="blue_illu")

pipeline.set_adapters(["dog", "style"], adapter_weights=[1.0, 1.0])

# %%

handler = sa_handler_guni.Handler(pipeline)
sa_args = sa_handler_guni.StyleAlignedArgs(share_group_norm=False,
                                      share_layer_norm=False,
                                      share_attention=True,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=False,
                                      shared_score_scale=0.5,
                                     )

handler.register(sa_args, controller)

# %%
p = [
    'a woman surrounded by blue decoration on a white background in flat cartoon illustration style',
    'a ohwx dog surrounded by blue decoration on a white background in flat cartoon illustration style'
]
g_cpu = torch.Generator(device='cpu')
g_cpu.manual_seed(1)

latents = torch.randn(len(p), 4, 128, 128, device='cpu', generator=g_cpu,
                      dtype=pipeline.unet.dtype,).to('cuda:0')
images = pipeline(p, latents=latents, controller=controller).images
result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
for i, image in enumerate(images):
    result_image.paste(image, (i * 1024, 0))
result_image.show()

# %% 24.1.26.
import numpy as np
import ptp_utils
def aggregate_attention(prompts, controller, res: int, from_where, select: int):
    out = []
    attention_maps = controller.get_average_attention()
    num_pixels = res ** 2
    out_dict = {}
    for location in from_where:
        location = f"{location}_cross"
        out_dict[location] = {}
        for case in attention_maps[location]:
            out_dict[location][case] = []
            for item in attention_maps[location][case]:
                if item.shape[2] == num_pixels:
                    attn_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out_dict[location][case].append(attn_maps)
    for location in out_dict:
        for case in out_dict[location]:
            for s in out_dict[location][case]:
                print(s.shape)
            temp = torch.cat(out_dict[location][case], dim=0)
            print(temp.shape)
            temp = temp.sum(0) / temp.shape[0]
            print(f"{location} | {case} | {temp.shape}")
            out_dict[location][case] = temp.cpu()
    return out_dict

def show_cross_attention(pipeline,
                            prompts, 
                            controller: AttentionStore, 
                            res: int, 
                            from_where, 
                            select: int = 0):
    tokens = pipeline.tokenizer.encode(prompts[select])
    decoder = pipeline.tokenizer.decode
    attention_dict = aggregate_attention(prompts, controller, res, from_where, select)
    for location in attention_dict:
        print(f"location: {location}")
        for case in attention_dict[location]:
            images = []
            print(f"case: {case}")
            attention_maps = attention_dict[location][case]
            for i in range(len(tokens)):
                image = attention_maps[:, :, i]
                image = 255 * image / image.max()
                image = image.unsqueeze(-1).expand(*image.shape, 3)
                image = image.numpy().astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((256, 256)))
                image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
                images.append(image)
            ptp_utils.view_images(np.stack(images, axis=0))
    return attention_dict
# %%
attention_dict = show_cross_attention(pipeline, p, feature_controller, 32, ('down', 'up'), 1)

# %%
original_module_dict = dict(pipeline.unet.named_modules())
module_dict = dict(pipeline.unet.named_modules())

lora_name = ['lora_A', 'lora_B']
adapter_name = ['dog', 'matt']
n = 0
content_A_and_B = {}
style_A_and_B = {}
for key, module in module_dict.items():
    if ('lora_A' in key or 'lora_B' in key) and (adapter_name[0] in key): # content
        for p in module.parameters():
            content_A_and_B[key] = p
    elif ('lora_A' in key or 'lora_B' in key) and (adapter_name[1] in key): # style
        for p in module.parameters():
            style_A_and_B[key] = p

content_AB_dict = {}
for name, module in content_A_and_B.items():
    if 'lora_A' in name:
        name_A = name
        module_A = module
    elif ('lora_B' in name) and (name.split('.')[:-2] == name_A.split('.')[:-2]):
        module_B = module
        module_AB = torch.mm(module_A.T, module_B.T)
        content_AB_dict['.'.join(name.split('.')[:-2])] = module_AB

style_AB_dict = {}
for name, module in style_A_and_B.items():
    if 'lora_A' in name:
        name_A = name
        module_A = module
    elif ('lora_B' in name) and (name.split('.')[:-2] == name_A.split('.')[:-2]):
        module_B = module
        module_AB = torch.mm(module_A.T, module_B.T)
        style_AB_dict['.'.join(name.split('.')[:-2])] = module_AB

# %%
import torch.nn.functional as nnf
n = 0
for i, key in enumerate(content_AB_dict):
    original_content_AB = content_AB_dict[key]
    original_content_AB_size = torch.norm(original_content_AB, p=2, dim=0)
    original_style_AB = style_AB_dict[key]
    original_style_AB_size = torch.norm(original_style_AB, p=2, dim=0)
    
    content_bigger_than_style = torch.zeros_like(original_content_AB_size)
    content_bigger_than_style[original_content_AB_size > original_style_AB_size] = 1
    
    content = nnf.normalize(original_content_AB, dim=0)
    style = nnf.normalize(original_style_AB, dim=0)
    cos_sim = (content * style).sum(dim=0)
    over_ratio = torch.zeros_like(cos_sim)
    over_ratio[(cos_sim >= 0.01) | (cos_sim <= -0.01)] = 1
    
    content_B_coef = torch.ones_like(content_bigger_than_style)
    content_B_coef[(over_ratio == 1) & (content_bigger_than_style == 0)] = 0
    
    style_B_coef = torch.ones_like(content_bigger_than_style)
    style_B_coef[(over_ratio == 1) & (content_bigger_than_style == 1)] = 0
    
    back_to_content_key = key + f'.lora_B.{adapter_name[0]}'
    back_to_style_key = key + f'.lora_B.{adapter_name[1]}'

    prev_content_weight = module_dict[back_to_content_key].weight
    masked_content_weight = torch.nn.Parameter(content_B_coef.unsqueeze(1) * prev_content_weight)
    module_dict[back_to_content_key].weight = masked_content_weight
    
    prev_style_weight = module_dict[back_to_style_key].weight
    masked_style_weight = torch.nn.Parameter(style_B_coef.unsqueeze(1) * prev_style_weight)
    module_dict[back_to_style_key].weight = masked_style_weight
    """
    print(torch.count_nonzero(over_ratio))
    print(over_ratio.shape[0] - torch.count_nonzero(over_ratio))
    print(key)
    print(cos_sim.shape)
    plt.figure(figsize=(10, 6))
    plt.hist(cos_sim.detach().cpu().numpy(), bins=100)
    plt.show()
    """
# %%

for name, module in pipeline.unet.named_modules():
    if 'lora_B.' in name:
        module.weight = module_dict[name].weight

# %%
handler.show_self_attention_comp_three_rgb(p, controller, res=64, from_where=("up", "down"), max_com=3, select=1)
# %%
handler.show_self_attention_comp(p, controller, res=64, from_where=("up", "down"), max_com=10, select=1)


# %%
import numpy as np
import ptp_utils
from sklearn.decomposition import PCA

class Show():
    def show_self_attention_comp(self, 
                                    prompts, 
                                    controller: AttentionStore, 
                                    res: int, 
                                    from_where, 
                                    max_com=10, 
                                    select: int = 0):
        attention_maps = self.aggregate_attention(prompts, controller, res, from_where, is_cross=False, select=select).numpy().astype(np.float32).reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        images = []
        for i in range(max_com):
            image = vh[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8) # [64, 64, 3]
            image = Image.fromarray(image).resize((256, 256))
            image = np.array(image) # [64, 64, 3]
            images.append(image)
        #return np.concatenate(images, axis=1)
        ptp_utils.view_images(np.concatenate(images, axis=1))


    def show_self_attention_comp_three_rgb(self, 
                                 prompts, 
                                 controller: AttentionStore, 
                                 res: int, 
                                 from_where, 
                                 max_com=3, 
                                 select: int = 0):
        attention_maps = self.aggregate_attention(prompts, controller, res, from_where, is_cross=False, select=select).numpy().astype(np.float32).reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        #image = vh[:max_com].reshape(res, res, max_com) # [64, 64, 3]
        V = vh[:max_com]
        images = []
        for i in range(max_com):
            image = V[i].reshape(res, res)
            image = image - image.min()
            image = 255 * image / image.max()
            image = Image.fromarray(image.astype(np.uint8)).resize((256, 256))
            image = np.array(image)
            images.append(image)
        ptp_utils.view_images(np.stack(images, axis=-1))


    def show_self_attention_pca(self,
                                prompts,
                                controller,
                                res,
                                from_where):
        def aggregate_attention_for_pca(prompts, attention_store: AttentionStore, res: int, from_where):
            out_first = []
            out_second = []
            attention_maps = attention_store.get_average_attention()
            num_pixels = res ** 2
            for location in from_where:
                for item in attention_maps[f"{location}_self"]:
                    if item.shape[2] == num_pixels:
                        attn_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])
                        out_first.append(attn_maps[0])
                        out_second.append(attn_maps[1])
            out_first = torch.cat(out_first, dim=0)
            out_second = torch.cat(out_second, dim=0)
            out_first = out_first.sum(0) / out_first.shape[0]
            out_second = out_second.sum(0) / out_second.shape[0]
            return out_first.cpu(), out_second.cpu()
        
        attn_maps_first, attn_maps_second = aggregate_attention_for_pca(prompts, 
                                                     controller, 
                                                     res, 
                                                     from_where)
        attn_maps_first = attn_maps_first.numpy().astype(np.float32).reshape((res ** 2, res ** 2))
        attn_maps_second = attn_maps_second.numpy().astype(np.float32).reshape((res ** 2, res ** 2))

        for x in [attn_maps_first, attn_maps_second]:
            pca = PCA(n_components=3)
            pca.fit(x)
            decom_first = pca.transform(attn_maps_first).reshape(res, res, 3)
            decom_second = pca.transform(attn_maps_second).reshape(res, res, 3)
            images = []
            for decom in [decom_first, decom_second]:
                decom = decom - decom.min()
                decom = 255 * decom / decom.max()
                decom = Image.fromarray(decom.astype(np.uint8)).resize((256, 256))
                decom = np.array(decom)
                images.append(decom)
            ptp_utils.view_images(np.concatenate(images, axis=1))

        return decom_first, decom_second

handler.show_self_attention_comp = Show.show_self_attention_comp
handler.show_self_attention_comp_three_rgb = Show.show_self_attention_comp_three_rgb
handler.show_self_attention_pca = Show.show_self_attention_pca

# %%
handler.show_self_attention_comp_three_rgb(handler, p, controller, res=64, from_where=("up", "down"), max_com=3, select=1)

# %%
handler.show_self_attention_comp(handler, p, controller, res=64, from_where=("up", "down"), max_com=10, select=1)

# %%
a = handler.show_self_attention_pca(handler, p, controller, res=64, from_where=("up", "down"))

# %%
result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
for i, image in enumerate(images):
    result_image.paste(image, (i * 1024, 0))
result_image.show()

# %%
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
        style_prompt,
        f"a ohwx dog wearing a hat{style_description}",
        f"a ohwx dog with a crown{style_description}",
        f"a ohwx dog riding a bicycle{style_description}",
    ]

    sets_of_prompts3 = [
        style_prompt,
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
best_seed = 2
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
        style_prompt,
        f"{style_description}a ohwx dog wearing a hat",
        f"{style_description}a ohwx dog with a crown",
        f"{style_description}a ohwx dog riding a bicycle",
    ]

    sets_of_prompts3 = [
        style_prompt,
        f"{style_description}a ohwx dog sleeping",
        f"{style_description}a ohwx dog in a boat",
        f"{style_description}a ohwx dog driving a car",
    ]
    return [sets_of_prompts1, sets_of_prompts2, sets_of_prompts3]

style_description = 'a matt black sculpture of '
style_prefix = f'gorilla face'
#style_description = ', macro photo, 3d game asset'
#style_prefix = f'a toy train'


set_p = set_of_prompts(style_description, style_prefix)
best_seed = 0

# %% DIFFERENT SEED
seeds_list = [[best_seed, 10, 20, 30], [best_seed, 40, 50, 60], [best_seed, 70, 80, 90]]
all_images = Image.new('RGB', (1024 * 10, 1024), color='white')
n = 0
for k in range(len(set_p)):
    p = set_p[k]
    if k != 0:
        print('\n'.join(p[1:]))
    else:
        print('\n'.join(p))
    latents = torch.tensor([], dtype=pipeline.unet.dtype,)
    for seed in seeds_list[k]:
        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(seed)
        temp = torch.randn(1, 4, 128, 128, device='cpu', generator=g_cpu,
                                dtype=pipeline.unet.dtype,)
        latents = torch.cat([latents, temp], dim=0)
    print(latents.shape)

    images = pipeline(p, latents=latents).images

    if k != 0:
        images = images[1:]
    result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
    for i, image in enumerate(images):
        result_image.paste(image, (i * 1024, 0))
        all_images.paste(image, (n * 1024, 0))
        n += 1
    result_image.show()
all_images.show()
# %%

# %% SAME SEEDS
all_images = Image.new('RGB', (1024 * 10, 1024), color='white')
n = 0
for k in range(len(set_p)):
    p = set_p[k]
    if k != 0:
        print('\n'.join(p[1:]))
    else:
        print('\n'.join(p))
    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(best_seed)
    latents = torch.randn(1, 4, 128, 128, device='cpu', generator=g_cpu,
                        dtype=pipeline.unet.dtype,).expand(len(p), 4, 128, 128).to('cuda:0')

    images = pipeline(p, latents=latents).images

    if k != 0:
        images = images[1:]
    result_image = Image.new('RGB', (1024 * len(images), 1024), color='white')
    for i, image in enumerate(images):
        result_image.paste(image, (i * 1024, 0))
        all_images.paste(image, (n * 1024, 0))
        n += 1
    result_image.show()
all_images.show()

# %%
p = [
    'a chocolate cookie'
]
g_cpu = torch.Generator(device='cpu')
g_cpu.manual_seed(1)

latents = torch.randn(len(p), 4, 128, 128, device='cpu', generator=g_cpu,
                      dtype=pipeline.unet.dtype,).to('cuda:0')
images = pipeline(p, latents=latents).images
images[0].show()
# %%
pipeline.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1