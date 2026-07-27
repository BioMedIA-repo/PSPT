"""CLAM-MB head used by PSPT."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, class_count, dropout=True):
        super().__init__()
        branch_a = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        branch_b = [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        if dropout:
            branch_a.append(nn.Dropout(0.25))
            branch_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*branch_a)
        self.attention_b = nn.Sequential(*branch_b)
        self.attention_c = nn.Linear(hidden_dim, class_count)

    def forward(self, features):
        gated = self.attention_a(features) * self.attention_b(features)
        return self.attention_c(gated), features


class CLAMMBInferenceHead(nn.Module):
    def __init__(self, size, class_count=3):
        super().__init__()
        self.n_classes = class_count
        self.attention_net = nn.Sequential(
            nn.Linear(size[0], size[1]),
            nn.ReLU(),
            nn.Dropout(0.25),
            GatedAttention(
                input_dim=size[1],
                hidden_dim=size[2],
                class_count=class_count,
                dropout=True,
            ),
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(size[1], 1) for _ in range(class_count)]
        )

    def forward(self, features):
        attention, hidden = self.attention_net(features)
        attention = F.softmax(attention.transpose(1, 0), dim=1)
        pooled = attention @ hidden
        logits = torch.empty(
            1,
            self.n_classes,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        for class_index in range(self.n_classes):
            logits[0, class_index] = self.classifiers[class_index](
                pooled[class_index]
            )
        return logits, F.softmax(logits, dim=1)
