import torch
import torch.nn as nn
import torch.optim as optim

def build_optimizer(cfg, params_or_model):
    opt_cfg = cfg.train_module.optimizer

    # 兼容两种情况：
    # 1) 传的是 model        -> 用 model.parameters()
    # 2) 传的是参数/参数组    -> 直接用
    if hasattr(params_or_model, "parameters"):
        params = params_or_model.parameters()
    else:
        params = params_or_model

    if opt_cfg.name == "adam":
        return torch.optim.Adam(
            params,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    elif opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")