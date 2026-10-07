import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_clam import Attn_Net, Attn_Net_Gated, initialize_weights


class CLAMSurvivalHead(nn.Module):


    def __init__(self, size=(1024, 256, 128), dropout=True, n_bins=4, gate=True):
        super().__init__()
        layers = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        attention_cls = Attn_Net_Gated if gate else Attn_Net
        layers.append(
            attention_cls(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        )
        self.attention_net = nn.Sequential(*layers)
        self.rho = nn.Sequential(
            nn.Linear(size[1], size[2]),
            nn.ReLU(),
            nn.Dropout(0.25 if dropout else 0.0),
        )
        self.classifier = nn.Linear(size[2], n_bins)
        self.n_bins = n_bins
        initialize_weights(self)

    def forward(self, features, attention_only=False):
        attention, hidden = self.attention_net(features)
        attention = attention.transpose(1, 0)
        if attention_only:
            return attention
        attention_raw = attention
        attention = F.softmax(attention, dim=1)
        pooled = torch.mm(attention, hidden)
        logits = self.classifier(self.rho(pooled))
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=1)
        risk = -survival.sum(dim=1)
        return hazards, survival, risk, attention_raw
