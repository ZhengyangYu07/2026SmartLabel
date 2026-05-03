"""
FlexMatch Adapter

提供一个轻量适配器，允许平台使用已有的 PyTorch DataLoader 和自定义模型直接调用 FlexMatch 的训练/评估接口。

用法示例：
from flexmatch_adapter import FlexMatchAdapter
adapter = FlexMatchAdapter(model=my_model, num_classes=..., device='cuda')
adapter.set_data_loaders(train_lb_loader, train_ulb_loader, eval_loader)
adapter.set_optimizer(optimizer, scheduler)
adapter.train(args)

注意：Adapter 假设传入的 DataLoader 产出格式与原实现一致：
- 有标注的 loader 输出: (idx, x_lb, y_lb)
- 无标注的 loader 输出: (idx, x_ulb_w, x_ulb_s)
评估 loader 输出: (idx, x, y)
如果平台的 DataLoader 不一致，请在平台端包装为上述格式。
"""
import types
import torch
# Defer importing FlexMatch heavy dependencies until adapter is actually constructed


class FlexMatchAdapter:
    def __init__(self, model, num_classes, device='cuda', **kwargs):
        """使用已有 `model` 实例构造 FlexMatch。

        model: 已初始化的 nn.Module（未包装到 DataParallel）
        num_classes: 类别数
        device: 'cuda' 或 'cpu'
        """
        self.device = device

        def net_builder(num_classes=None):
            # FlexMatch expects a callable that returns a freshly constructed network given num_classes.
            # 为兼容性，返回传入的 model 实例（注意：外部应保证训练/评估前将 model 放到正确 device）。
            return model

        # 默认超参可由外部 args 覆盖
        from models.flexmatch.flexmatch import FlexMatch
        self.fm = FlexMatch(net_builder, num_classes,
                    kwargs.get('ema_m', 0.999),
                    kwargs.get('T', 0.5),
                    kwargs.get('p_cutoff', 0.95),
                    kwargs.get('lambda_u', 1.0),
                    hard_label=kwargs.get('hard_label', True),
                    num_eval_iter=kwargs.get('num_eval_iter', 1000),
                    tb_log=kwargs.get('tb_log', None),
                    logger=kwargs.get('logger', None))

        self.model = model

    def set_data_loaders(self, train_lb_loader, train_ulb_loader, eval_loader):
        loader_dict = {'train_lb': train_lb_loader, 'train_ulb': train_ulb_loader, 'eval': eval_loader}
        self.fm.set_data_loader(loader_dict)
        self.fm.set_dset(getattr(self, 'ulb_dataset', None))

    def set_unlabeled_dataset(self, ulb_dataset):
        # 如果需要，将 unlabeled dataset 直接传入（供内部记录 selected_label 长度使用）
        self.fm.set_dset(ulb_dataset)
        self.ulb_dataset = ulb_dataset

    def set_optimizer(self, optimizer, scheduler=None):
        self.fm.set_optimizer(optimizer, scheduler)

    def train(self, args):
        # args 是 argparse.Namespace 或者具有相同行为的对象
        # 确保 model/ema 在正确 device
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()
        return self.fm.train(args)

    def evaluate(self, eval_loader=None, args=None):
        return self.fm.evaluate(eval_loader=eval_loader, args=args)

    def save_model(self, save_name, save_path):
        return self.fm.save_model(save_name, save_path)

    def load_model(self, load_path):
        return self.fm.load_model(load_path)
