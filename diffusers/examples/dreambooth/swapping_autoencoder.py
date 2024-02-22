import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_features, dropout_rate):
        super().__init__()
        down_features = in_features // 2
        general_features = down_features // 2
        control_features = down_features // 2
        self.to_down = nn.Linear(in_features, down_features)
        self.to_general = nn.Linear(down_features, general_features)
        self.to_control = nn.Linear(down_features, control_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        down_embeddings = self.to_down(x)
        general_embeddings = self.dropout(self.to_general(down_embeddings))
        control_embeddings = self.dropout(self.to_control(down_embeddings))
        return down_embeddings, general_embeddings, control_embeddings


class Decoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        sum_features = in_features // 2
        self.to_sum = nn.Linear(sum_features, sum_features)
        self.to_up = nn.Linear(sum_features, in_features)
    
    def forward(self, down, general, control):
        general = general[0].unsqueeze(0).expand(2, *general.shape[1:])
        x = torch.cat([general, control], dim=-1)
        x = self.to_sum(x) + down
        x = self.to_up(x)
        return x


class SwappingAutoencoder(nn.Module):
    def __init__(self, in_features, dropout_rate=0.2):
        super().__init__()
        self.E = Encoder(in_features, dropout_rate)
        self.D = Decoder(in_features)
    
    def forward(self, x, *args, **kwargs):
        down, general, control = self.E(x)
        x = self.D(down, general, control)
        return x
