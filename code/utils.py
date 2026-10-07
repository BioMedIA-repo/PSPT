"""Minimal runtime parameter/tensor helpers."""
import json
import os
import torch
from torch import nn

def save_parameters(args):
    folder_path = os.path.join(args.output_dir, args.run_name)
    os.makedirs(folder_path, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(folder_path, "parameters.json"), "w") as f:
        json.dump(
            {n: str(args_dict[n]) for n in args_dict},
            f,
            indent=4
        )

class switch_dim(nn.Module):
    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        return x
