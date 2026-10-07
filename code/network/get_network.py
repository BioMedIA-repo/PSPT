def set_transfer_type(model, transfer_type):
    if transfer_type != 'pspt':
        raise ValueError('This compact training distribution supports transfer_type=pspt')
    for name, parameter in model.named_parameters():
        enabled=any(key in name for key in ('prompt','scpm','fprd'))
        if 'scpm' in name and not model.scpm_enabled:enabled=False
        if 'fprd' in name and not model.fprd_enabled:enabled=False
        parameter.requires_grad=enabled
    return model


def parameters(args):
    return dict(num_tokens=args.num_prompt_tokens,drop_out=args.prompt_dropout,
        scpm_latent_dim=args.scpm_latent_dim,scpm_enabled=not args.disable_scpm,
        fprd_enabled=args.enable_fprd,fprd_neighbors=args.fprd_neighbors,
        fprd_temperature=args.fprd_temperature,fprd_time_max=args.fprd_time_max,
        fprd_iterations=args.fprd_iterations)


def get_uni_peft_model(args):
    import torch
    from .uni_pspt import PSPTBackbone
    model=PSPTBackbone(**parameters(args))
    state=torch.load(args.load_backbone_weight,map_location='cpu',weights_only=True)
    model.vit.load_state_dict(state,strict=True)
    return set_transfer_type(model,args.transfer_type)


def get_conch_peft_model(args):
    from .conch_pspt import CONCH_PSPT
    model=CONCH_PSPT(checkpoint_path=args.load_backbone_weight,
                     image_size=args.backbone_image_size,**parameters(args))
    return set_transfer_type(model,args.transfer_type)


def get_plip_peft_model(args):
    from .plip_pspt import PLIP_PSPT
    model=PLIP_PSPT(checkpoint_path=args.load_backbone_weight,
                    image_size=args.backbone_image_size,**parameters(args))
    return set_transfer_type(model,args.transfer_type)
