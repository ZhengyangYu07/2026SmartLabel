import argparse
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import random
import shutil
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import get_cifar100, get_cifar10
from dataset.custom_dataset import CustomDataset, UnlabeledDataset
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


# 设置默认值
def set_default_attr(args, name, default):
    if not hasattr(args, name) or getattr(args, name) is None:
        setattr(args, name, default)


def parse_args_from_json(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    args = argparse.Namespace(**config_dict)
    return args, config_dict


def update_json_config(json_path, updated_config):
    with open(json_path, 'w') as f:
        json.dump(updated_config, f, indent=2)
    print(f"[INFO] Updated config saved to {json_path}")


def auto_set_num_classes(args, config_dict):
    if not os.path.exists(args.label_csv):
        raise FileNotFoundError(f"Label CSV not found: {args.label_csv}")

    df = pd.read_csv(args.label_csv)
    if 'label' not in df.columns:
        raise ValueError(f"'label' column not found in {args.label_csv}")

    num_classes = df['label'].nunique()
    args.num_classes = num_classes
    config_dict['num_classes'] = num_classes
    print(f"[INFO] Detected {num_classes} unique classes in label CSV.")
    return args, config_dict


def get_custom_dataset(args, data_dir):
    import pandas as pd
    from torchvision import transforms
    # 加载标签映射关系
    label_csv = args.label_csv
    image_dir = args.image_dir

    # 从CSV读取标签到ID的映射（核心修正点）
    df = pd.read_csv(label_csv)
    unique_labels = df["label"].unique().tolist()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}  # 修复空映射问题[1](@ref)

    # 保存映射字典（输出到结果目录）
    os.makedirs(args.out, exist_ok=True)  # 确保目录存在[6](@ref)
    with open(os.path.join(args.out, 'label2id.json'), 'w') as f:
        json.dump(label2id, f, indent=4)
    print(f"标签映射表已保存至 {args.out} 目录")

    # 获取所有已标注的文件名
    labeled_files = set(df["filename"].tolist())

    # 获取未标注文件列表（修复逻辑错误）
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]  # 过滤图像文件
    unlabeled_files = [f for f in all_images if f not in labeled_files]  # 正确筛选未标注文件[2](@ref)

    # 定义数据增强策略
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()
    ])

    # 创建数据集实例
    labeled_dataset = CustomDataset(
        image_dir=image_dir,
        label_csv=label_csv,
        label2id=label2id,
        transform=weak_transform  # 添加数据增强[3](@ref)
    )

    unlabeled_dataset = UnlabeledDataset(
        image_dir=image_dir,
        unlabeled_files=unlabeled_files,  # 传入文件列表而非CSV路径[2](@ref)
        weak_transform=weak_transform,
        strong_transform=strong_transform
    )

    # 创建测试集（需拆分训练数据或单独指定测试路径）
    test_dataset = CustomDataset(
        image_dir=image_dir,
        label_csv=label_csv,
        label2id=label2id,
        transform=transforms.ToTensor(),  # 测试集只需基础转换
        # test=True
    )

    print(f"CSV标注文件数：{len(df)}，实际图像文件数：{len(all_images)}")
    print(f"未标注文件示例：{unlabeled_files[:5]}")

    return labeled_dataset, unlabeled_dataset, test_dataset


DATASET_GETTERS = {
    'cifar10': get_cifar10,  # 已有的例子
    'cifar100': get_cifar100,  # 已有的例子
    'custom': get_custom_dataset  # 新添加的 custom 数据集处理函数
}


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--config', type=str, default='./config.json', help='Path to config .json file')
    args_cmd = parser.parse_args()

    # 1. 加载 JSON 配置为 args 对象和 dict 对象
    args, config_dict = parse_args_from_json(args_cmd.config)

    # 2. 自动设置 num_classes（如未指定或需更新）
    args, config_dict = auto_set_num_classes(args, config_dict)

    # 3. 写回更新后的 JSON 文件
    update_json_config(args_cmd.config, config_dict)

    print(f"Using config from {args_cmd.config}")
    print(f"Dataset: {args.dataset}, Model: {args.arch}, Labeled: {args.num_labeled}")

    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}"
    )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        import os
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        set_default_attr(args, 'num_classes', 10)
        if args.arch == 'wideresnet':
            set_default_attr(args, 'model_depth', 28)
            set_default_attr(args, 'model_width', 2)
        elif args.arch == 'resnext':
            set_default_attr(args, 'model_cardinality', 4)
            set_default_attr(args, 'model_depth', 28)
            set_default_attr(args, 'model_width', 4)

    elif args.dataset == 'cifar100':
        set_default_attr(args, 'num_classes', 100)
        if args.arch == 'wideresnet':
            set_default_attr(args, 'model_depth', 28)
            set_default_attr(args, 'model_width', 8)
        elif args.arch == 'resnext':
            set_default_attr(args, 'model_cardinality', 8)
            set_default_attr(args, 'model_depth', 29)
            set_default_attr(args, 'model_width', 64)

    elif args.dataset == 'custom':
        set_default_attr(args, 'num_classes', 100)
        if args.arch == 'wideresnet':
            set_default_attr(args, 'model_depth', 28)
            set_default_attr(args, 'model_width', 8)
        elif args.arch == 'resnext':
            set_default_attr(args, 'model_cardinality', 8)
            set_default_attr(args, 'model_depth', 29)
            set_default_attr(args, 'model_width', 64)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # if args.amp:
    #     from apex import amp
    #     model, optimizer = amp.initialize(
    #         model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    # if args.amp:
    #     from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                # inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                # inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                inputs_x, targets_x = next(labeled_iter)

            try:
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()

            loss = Lx + args.lambda_u * Lu

            # if args.amp:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    main()
