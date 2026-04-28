# https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/utils/ema.py
import torch.nn as nn


class EMA(object):
    """
    Usage:
        model = ResNet(config)
        model_ema = ResNet(config)
        ema = EMA(model, alpha=0.99)
        ... # train an epoch
        ema.update_params(model)
        ema.apply_shadow(model_ema)
    """
    def __init__(self, model: nn.Module, alpha: float=0.999):
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]
        self.alpha = alpha

    def init_params(self, model: nn.Module) -> None:
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.param_keys = [k for k, _ in model.named_parameters()]

    def update_params(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(self.alpha * self.shadow[name] + (1 - self.alpha) * state[name])

    def apply_shadow(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
