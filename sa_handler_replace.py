# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from diffusers.models import attention_processor
import einops
import ptp_utils
from ptp_utils import AttentionStore
from typing import List
from PIL import Image
from sklearn.decomposition import PCA

T = torch.Tensor


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True,
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.
    shared_score_shift: float = 0.
    only_self_level: float = 0.


def expand_first(feat: T, scale=1.,) -> T:
    b = feat.shape[0] # 8
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1) # torch.Size([2, 1, 10, 1, 64]) -> 8개 중 0번째, 4번째 stack. 첫 번째 prompt의 negative/positive prompt
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:]) # torch.Size([2, 4, 10, 1, 64])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape) # torch.Size([8, 10, 4096, 64])


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale) # torch.Size([8, 10, 4096, 64])
    return torch.cat((feat, feat_style), dim=dim) # torch.Size([8, 10, 8192, 64])

def concat_first_guni(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale) # torch.Size([8, 10, 4096, 64])
    return torch.cat((feat, feat_style * 0.5), dim=dim) # torch.Size([8, 10, 8192, 64])


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt() # torch.Size([8, 10, 1, 64])
    feat_mean = feat.mean(dim=-2, keepdims=True) # torch.Size([8, 10, 1, 64])
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat) # 둘 다 torch.Size([8, 10, 1, 64])
    feat_style_mean = expand_first(feat_mean) # torch.Size([8, 10, 1, 64])
    feat_style_std = expand_first(feat_std) # torch.Size([8, 10, 1, 64])
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


def adain_guni(feat: T) -> T:
    """
    feat_mean, feat_std = calc_mean_std(feat) # 둘 다 torch.Size([8, 10, 1, 64])
    feat_style_mean = expand_first(feat_mean) # torch.Size([8, 10, 1, 64])
    feat_style_std = expand_first(feat_std) # torch.Size([8, 10, 1, 64])
    feat = (feat - feat_mean) / feat_std
    feat = feat * 0.5 * (feat_style_std + feat_std) + 0.5 * (feat_style_mean + feat_mean)
    """
    return feat

class DefaultAttentionProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        if hasattr(nnf, "scaled_dot_product_attention"):
            self.processor = attention_processor.AttnProcessor2_0()
        else:
            self.processor = attention_processor.AttnProcessor()                

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)


class SharedAttentionProcessor_guni(DefaultAttentionProcessor):

    def __init__(self, style_aligned_args: StyleAlignedArgs, controller: AttentionStore, place_in_unet: str, is_self_attention: bool, is_attn_store: bool):
        super().__init__()
        self.share_attention = style_aligned_args.share_attention
        self.adain_queries = style_aligned_args.adain_queries
        self.adain_keys = style_aligned_args.adain_keys
        self.adain_values = style_aligned_args.adain_values
        self.full_attention_share = style_aligned_args.full_attention_share
        self.shared_score_scale = style_aligned_args.shared_score_scale
        self.shared_score_shift = style_aligned_args.shared_score_shift
        
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.is_self_attention = is_self_attention
        self.is_attn_store = is_attn_store
        self.switch_condition_ = None

    @torch.no_grad()
    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        if self.full_attention_share:
            b, n, d = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, '(k b) n d -> k (b n) d', k=2)
            hidden_states = super().__call__(attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
                                             attention_mask=attention_mask, **kwargs)
            hidden_states = einops.rearrange(hidden_states, 'k (b n) d -> (k b) n d', n=n)
        else:
            hidden_states = self.shared_call(attn, hidden_states, hidden_states, attention_mask, **kwargs)

        return hidden_states # torch.Size([8, 4096, 640])


    def shifted_scaled_dot_product_attention(self, attn: attention_processor.Attention, query: T, key: T, value: T) -> T:
        logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
        logits[:, :, :, query.shape[2]:] += self.shared_score_shift
        probs = logits.softmax(-1)
        return torch.einsum('bhqk,bhkd->bhqd', probs, value)

    @torch.no_grad()
    def scaled_dot_product_attention_guni(self, attn: attention_processor.Attention, query: T, key: T, value: T) -> T:
        logits = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
        logits[:, :, :, query.shape[2]:] += self.shared_score_shift
        probs = logits.softmax(-1) # torch.Size([4, 10, 4096, 8192])
        #probs_orig = probs.clone()
        """
        if self.is_attn_store:
            b = probs.shape[0]
            self.controller(torch.chunk(probs, 2, dim=-1)[0][b // 2:], self.is_self_attention, self.place_in_unet) # chunk: torch.Size([2, 10, 4096, 4096]) (예상)
        """
        return torch.einsum('bhqk,bhkd->bhqd', probs, value)

    @torch.no_grad()
    def shared_call(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) # torch.Size([8, 4096, 640])
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # torch.Size([8, 10, 4096, 64])
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # if self.step >= self.start_inject:
        if self.adain_queries:
            query = adain(query) # torch.Size([8, 10, 4096, 64])
        if self.adain_keys:
            key = adain(key)
        if self.adain_values:
            value = adain(value)
        #if self.share_attention and query.shape[2] == 4096 and self.place_in_unet == 'up':
        #if self.share_attention and self.place_in_unet == 'up':
        if self.share_attention:
            if (self.switch_condition_['how'] == 'equal' and query.shape[2] == 4096 and self.place_in_unet in self.switch_condition_['place_in_unet']) or (self.switch_condition_['how'] == 'smaller' and query.shape[2] < 4096 and self.place_in_unet in self.switch_condition_['place_in_unet']):
                key = expand_first(key, self.shared_score_scale)
                value = expand_first(value, self.shared_score_scale)
                if self.shared_score_shift != 0:
                    hidden_states = self.shifted_scaled_dot_product_attention(attn, query, key, value,)
                else:
                    hidden_states = self.scaled_dot_product_attention_guni(attn, query, key, value)
                    """
                    hidden_states = nnf.scaled_dot_product_attention(
                        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    ) # torch.Size([8, 10, 4096, 64])
                    """
            else:
                #hidden_states = nnf.scaled_dot_product_attention(
                #    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                #)
                ### guni 240226
                hidden_states = torch.einsum('bhqd,bhkd->bhqk', query, key) * attn.scale
                hidden_states = hidden_states.softmax(-1)
                hidden_states = torch.einsum('bhqk,bhkd->bhqd', hidden_states, value)


        # hidden_states = adain(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # torch.Size([8, 4096, 640])
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def _get_switch_vec(total_num_layers, level):
    if level == 0:
        return torch.zeros(total_num_layers, dtype=torch.bool)
    if level == 1:
        return torch.ones(total_num_layers, dtype=torch.bool)
    to_flip = level > .5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total_num_layers)
    vec = torch.arange(total_num_layers)
    vec = vec % (total_num_layers // num_switch)
    vec = vec == 0
    if to_flip:
        vec = ~vec
    return vec


def init_attention_processors(pipeline: StableDiffusionXLPipeline, style_aligned_args: StyleAlignedArgs | None = None, controller: AttentionStore | None = None):
    attn_procs = {}
    cross_att_count = 0
    unet = pipeline.unet
    number_of_self, number_of_cross = 0, 0
    num_self_layers = len([name for name in unet.attn_processors.keys() if 'attn1' in name])
    if style_aligned_args is None:
        only_self_vec = _get_switch_vec(num_self_layers, 1)
    else:
        only_self_vec = _get_switch_vec(num_self_layers, style_aligned_args.only_self_level)
    for i, name in enumerate(unet.attn_processors.keys()):
        is_self_attention = 'attn1' in name # self attention만
        if is_self_attention:
            number_of_self += 1
            if style_aligned_args is None or only_self_vec[i // 2]:
                attn_procs[name] = DefaultAttentionProcessor()
            else:
                if name.startswith("down_blocks"):
                    place_in_unet = "down"
                elif name.startswith("mid_block"):
                    place_in_unet = "mid"
                elif name.startswith("up_blocks"):
                    place_in_unet = "up"
                else:
                    continue
                is_attn_store = True if i % 4 == 0 else False
                cross_att_count += 1
                attn_procs[name] = SharedAttentionProcessor_guni(style_aligned_args, controller=controller, place_in_unet=place_in_unet, is_self_attention=is_self_attention, is_attn_store=is_attn_store)
        else:
            number_of_cross += 1
            attn_procs[name] = DefaultAttentionProcessor()

    unet.set_attn_processor(attn_procs)
    if controller:
        controller.num_att_layers = cross_att_count


def register_shared_norm(pipeline: StableDiffusionXLPipeline,
                         share_group_norm: bool = True,
                         share_layer_norm: bool = True, ):
    def register_norm_forward(norm_layer: nn.GroupNorm | nn.LayerNorm) -> nn.GroupNorm | nn.LayerNorm:
        if not hasattr(norm_layer, 'orig_forward'):
            setattr(norm_layer, 'orig_forward', norm_layer.forward)
        orig_forward = norm_layer.orig_forward

        def forward_(hidden_states: T) -> T:
            n = hidden_states.shape[-2]
            hidden_states = concat_first(hidden_states, dim=-2)
            hidden_states = orig_forward(hidden_states)
            return hidden_states[..., :n, :]

        norm_layer.forward = forward_
        return norm_layer

    def get_norm_layers(pipeline_, norm_layers_: dict[str, list[nn.GroupNorm | nn.LayerNorm]]):
        if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
            norm_layers_['layer'].append(pipeline_)
        if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
            norm_layers_['group'].append(pipeline_)
        else:
            for layer in pipeline_.children():
                get_norm_layers(layer, norm_layers_)

    norm_layers = {'group': [], 'layer': []}
    get_norm_layers(pipeline.unet, norm_layers)
    return [register_norm_forward(layer) for layer in norm_layers['group']] + [register_norm_forward(layer) for layer in
                                                                               norm_layers['layer']]


class Handler:

    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline
        self.norm_layers = []
    
    
    def register(self, style_aligned_args: StyleAlignedArgs, controller: AttentionStore = None):
        self.norm_layers = register_shared_norm(self.pipeline, style_aligned_args.share_group_norm,
                                                style_aligned_args.share_layer_norm)
        init_attention_processors(self.pipeline, style_aligned_args, controller)


    def remove(self):
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers = []
        init_attention_processors(self.pipeline, None, None)


    def aggregate_attention(self, prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
        out = []
        attention_maps = attention_store.get_average_attention()
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[2] == num_pixels:
                    attn_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                    out.append(attn_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.cpu()


    def show_cross_attention(self, 
                             prompts, 
                             controller: AttentionStore, 
                             res: int, 
                             from_where: List[str], 
                             select: int = 0):
        tokens = self.pipeline.tokenizer.encode(prompts[select])
        decoder = self.pipeline.tokenizer.decode
        attention_maps = self.aggregate_attention(prompts, controller, res, from_where, True, select)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        ptp_utils.view_images(np.stack(images, axis=0))
    
    
    def show_self_attention_comp(self, 
                                 prompts, 
                                 controller: AttentionStore, 
                                 res: int, 
                                 from_where: List[str], 
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
        ptp_utils.view_images(np.concatenate(images, axis=1))
        
    
    def show_self_attention_comp_three_rgb(self, 
                                 prompts, 
                                 controller: AttentionStore, 
                                 res: int, 
                                 from_where: List[str], 
                                 max_com=3, 
                                 select: int = 0):
        attention_maps = self.aggregate_attention(prompts, controller, res, from_where, is_cross=False, select=select).numpy().astype(np.float32).reshape((res ** 2, res ** 2))
        u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
        image = vh[:max_com].transpose(0, 1).reshape(res, res, max_com) # [64, 64, 3]
        image = image - image.min(axis=(0, 1))
        image = 255 * image / image.max(axis=(0, 1))
        image = Image.fromarray(image.astype(np.uint8)).resize((256, 256))
        image = np.array(image)
        ptp_utils.view_images(image)
    