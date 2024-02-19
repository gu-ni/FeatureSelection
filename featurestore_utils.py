import abc

import numpy as np
import torch
from typing import Union, Tuple, List, Dict, Optional
import torch.nn as nn
import torch.nn.functional as nnf


class FeatureControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_self_attention: bool, place_in_unet: str, attn_type: str, heads: int, scale: float, submodule_name: str):
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, features_dict, self_or_cross: bool, place_in_unet: str, attn_type: str, heads: int, scale: float, submodule_name: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.cur_step % 5 == 0:
                attn = self.forward(features_dict, self_or_cross, place_in_unet, attn_type, heads, scale, submodule_name)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            if self.cur_step % 5 == 0:
                self.between_steps()
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        


class FeatureStoreSelf(FeatureControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": {}, "mid_cross": {}, "up_cross": {},
                "down_self": {},  "mid_self": {},  "up_self": {}}

    @torch.no_grad()
    def forward(self, features_dict, self_or_cross: bool, place_in_unet: str, attn_type: str, heads: int):
        key = f"{place_in_unet}_{self_or_cross}"
        # 'self'일 때
        """
        # original_x랑 나머지 비교
        original_x = features_dict.pop('original_x', None)
        
        for name, value in features_dict.items():
            if self_or_cross == 'self':
                difference = torch.abs(original_x - value)
                mean = difference.mean(dim=-2)
                std = difference.std(dim=-2)
                if f'x-{name}' not in self.step_store[key]:
                    self.step_store[key][f'x-{name}'] = ([mean], [std])
                else:
                    self.step_store[key][f'x-{name}'][0].append(mean)
                    self.step_store[key][f'x-{name}'][1].append(std)
        """
        
        # 모든 feature 간 차이 조합
        if self_or_cross == 'self':
            features = [*features_dict.keys()]
            for i in range(len(features) - 1):
                for j in range(i + 1, len(features)):
                    difference = torch.abs(features_dict[features[i]] - features_dict[features[j]])
                    mean = difference.mean(dim=-2)
                    std = difference.std(dim=-2)
                    idx = f'{features[i]}-{features[j]}'
                    if idx not in self.step_store[key]:
                        self.step_store[key][idx] = ([mean], [std])
                    else:
                        self.step_store[key][idx][0].append(mean)
                        self.step_store[key][idx][1].append(std)


    def between_steps(self):
        # 'self'일 때
        if len(self.feature_store) == 0:
            self.feature_store = self.step_store
        else:
            for key in self.feature_store:
                for diff_name in self.feature_store[key]:
                    for i in range(len(self.feature_store[key][diff_name])):
                        for j in range(len(self.feature_store[key][diff_name][i])):
                            self.feature_store[key][diff_name][i][j] += self.step_store[key][diff_name][i][j]
        self.step_store = self.get_empty_store()


    def get_average_attention(self):
        # 'self'일 때
        average_attention = {key: {diff_name: tuple(list(map(lambda x: x / self.cur_step, mean_std)) 
                                                    for mean_std in self.feature_store[key][diff_name]) 
                                   for diff_name in self.feature_store[key]} 
                             for key in self.feature_store}
        return average_attention


    def reset(self):
        super(FeatureStoreSelf, self).reset()
        self.step_store = self.get_empty_store()
        self.feature_store = {}


    def __init__(self):
        super(FeatureStoreSelf, self).__init__()
        self.step_store = self.get_empty_store()
        self.feature_store = {}





class FeatureStoreCross(FeatureControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": {}, "mid_cross": {}, "up_cross": {},
                "down_self": {},  "mid_self": {},  "up_self": {}}

    @torch.no_grad()
    def forward(self, features_dict, self_or_cross: bool, place_in_unet: str, attn_type: str, heads: int, scale: float, submodule_name:str):
        key = f"{place_in_unet}_{self_or_cross}"
        # 'cross'일 때
        if self_or_cross == 'cross':
            if features_dict['original_x'].shape[-2] <= 32 ** 2:
                del features_dict['original_x']
                if attn_type == 'to_q':
                    self.query_dict = features_dict
                elif attn_type == 'to_k' and self.query_dict:
                    query_features_dict = self.query_dict
                    """
                    # 전체 다
                    query_result = query_features_dict.pop('base_layer_x', None)
                    for lora in query_features_dict.keys():
                        query_result += query_features_dict[lora]
                    del query_features_dict
                    
                    batch_size = query_result.shape[0]
                    inner_dim = query_result.shape[-1]
                    head_dim = inner_dim // heads
                    
                    query_result = query_result.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                    
                    key_lora = [*features_dict.keys()]
                    key_lora.remove('base_layer_x')
                    case_list = [['base_layer_x']] + [['base_layer_x', lora] for lora in key_lora] + [['base_layer_x'] + key_lora]
                    for case in case_list:
                        key_result = features_dict[case[0]]
                        for layer in case[1:]:
                            key_result += features_dict[layer]
                        key_result = key_result.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                        logits = torch.einsum('bhqd,bhkd->bhqk', query_result, key_result)
                        probs = logits.softmax(-1)
                        
                        case = '+'.join(case)
                        if case not in self.step_store[key]:
                            self.step_store[key][case] = [probs]
                        else:
                            self.step_store[key][case].append(probs)
                    
                    self.query_dict = None
                    """
                    # 같이
                    key_lora = [*features_dict.keys()]
                    key_lora.remove('base_layer_x')
                    case_list = [['base_layer_x']] + [['base_layer_x', lora] for lora in key_lora] + [['base_layer_x'] + key_lora]
                    for case in case_list[3:]:
                        query_result = query_features_dict[case[0]]
                        key_result = features_dict[case[0]]
                        for layer in case[1:]:
                            query_result += query_features_dict[layer]
                            key_result += features_dict[layer]
                    
                        batch_size = query_result.shape[0]
                        inner_dim = query_result.shape[-1]
                        head_dim = inner_dim // heads
                        
                        query_result = query_result.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                        key_result = key_result.view(batch_size, -1, heads, head_dim).transpose(1, 2)
                        b = query_result.shape[0]
                        query_result = query_result[b // 2:]
                        key_result = key_result[b // 2:]
                        logits = torch.einsum('bhqd,bhkd->bhqk', query_result, key_result) * scale
                        del key_result
                        probs = logits.softmax(-1)
                        # print(f"shape of probs is {probs.shape}")
                        del logits
                        
                        case = '+'.join(case)
                        if case not in self.step_store[key]:
                            self.step_store[key][case] = [probs]
                        else:
                            self.step_store[key][case].append(probs)
                        del probs
                    
                    self.query_dict = None
                    

    def between_steps(self):        
        # 'cross'일 때
        if len(self.feature_store) == 0:
            self.feature_store = self.step_store
        else:
            for key in self.feature_store:
                for case in self.feature_store[key]:
                    self.feature_store[key][case] += self.step_store[key][case]
        self.step_store = self.get_empty_store()


    def get_average_attention(self): # 웍스3에 있던 코드 (24.1.26.)
        # 'cross'일 때
        average_attention = {key: {case: list(map(lambda x: x / 5, array)) 
                                   for case, array in self.feature_store[key].items()}
                             for key in self.feature_store}
        return average_attention
    
    """
    def get_average_attention(self):
        # 'cross'일 때
        average_attention = {key: {case: list(map(lambda x: x / self.cur_step)) 
                                   for case in self.feature_store[key]}
                             for key in self.feature_store}
        return average_attention
    """
    
    def reset(self):
        super(FeatureStoreCross, self).reset()
        self.step_store = self.get_empty_store()
        self.feature_store = {}
        self.query_dict = None

    def __init__(self):
        super(FeatureStoreCross, self).__init__()
        self.step_store = self.get_empty_store()
        self.feature_store = {}
        self.query_dict = None