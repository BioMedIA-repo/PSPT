from functools import partial
import os
from pathlib import Path

import torch
from torchvision import transforms

IMAGENET=([.485,.456,.406],[.229,.224,.225])
CLIP=([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])


def get_encoder(model_name,target_img_size=224):
    weight=Path(os.environ['PSPT_ENCODER_WEIGHTS'])
    if model_name=='uni_v1':
        import timm
        model=timm.create_model('vit_large_patch16_224',init_values=1e-5,num_classes=0,dynamic_img_size=True)
        path=weight/'pytorch_model.bin' if weight.is_dir() else weight
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        mean,std=IMAGENET
    elif model_name=='conch_v1':
        from conch.open_clip_custom import create_model_from_pretrained
        path=weight/'pytorch_model.bin' if weight.is_dir() else weight
        model,_=create_model_from_pretrained('conch_ViT-B-16',str(path))
        model.forward=partial(model.encode_image,proj_contrast=False,normalize=False)
        mean,std=CLIP
    elif model_name=='plip':
        from transformers import CLIPModel
        class PLIPWrapper(torch.nn.Module):
            def __init__(self,path):
                super().__init__();self.model=CLIPModel.from_pretrained(path)
            def forward(self,x):return self.model.get_image_features(pixel_values=x)
        model=PLIPWrapper(str(weight.parent if weight.is_file() else weight))
        mean,std=CLIP
    else:raise ValueError(model_name)
    transform=transforms.Compose([transforms.Resize(target_img_size),transforms.ToTensor(),transforms.Normalize(mean,std)])
    return model,transform
