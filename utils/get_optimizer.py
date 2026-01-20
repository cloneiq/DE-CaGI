import torch

def get_optimizer(model):
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'vision_encoder' in name or 'language_encoder' in name:
            backbone_params.append(param)
            
        else:
            head_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": backbone_params, 
            "lr": 5e-6,
            "weight_decay": 0.05
        },
        {
            "params": head_params, 
            "lr": 1e-4,
            "weight_decay": 0.05
        }
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer