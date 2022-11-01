import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from dataset import MaskBaseDataset

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import f1_score  # from torchmetrics.classification import MulticlassF1Score
from sklearn.model_selection import StratifiedKFold
from importlib import import_module
from loss import create_criterion
import gc

gc.collect()
torch.cuda.empty_cache()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
class CutMix(object):
    def __init__(self, beta, cutmix_prob) -> None:
        super().__init__()
        self.beta = beta
        self.cutmix_prob = cutmix_prob
        
    def forward(self, images, labels):
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(images.size()[0]).cuda()
        label_1 = labels
        label_2 = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2-bbx1)*(bby2-bby1)/(images.size()[-1]*images.size()[-2]))
        
        return {'lam': lam, 'image': images, 'label_1': label_1, 'label_2': label_2}
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.model, args.criterion))  # 실험하는 하이퍼파라미터들로 채우기

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset = MaskBaseDataset(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- data_loader
    n_val = int(len(dataset) * 0.3)
    n_train = len(dataset) - n_val
    train_set, val_set = data.random_split(dataset, [n_train, n_val])
    
    # -- augmentation
    train_transform = A.Compose([
        A.CLAHE(p=0.5),
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.56, 0.524, 0.501), std=(0.233, 0.243, 0.246)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.56, 0.524, 0.501), std=(0.233, 0.243, 0.246)),
        ToTensorV2(),
    ])
    
    train_set.dataset.set_transform(train_transform)
    val_set.dataset.set_transform(val_transform)
    
    # cutmix ; 하이퍼파라미터 설정
    use_cutmix = True
    cutmix_prob = 0.3
    cutmix = CutMix(beta=1.0, cutmix_prob=cutmix_prob)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행
    #n_splits = 5
    #skf = StratifiedKFold(n_splits=n_splits)

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: ResNet34
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
#     f1_macro = MulticlassF1Score(num_classes=18, average='macro').to(device)
#     f1_micro = MulticlassF1Score(num_classes=18, average='micro').to(device)
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=1) # StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    # early_stopping : 8번의 epoch 연속으로 loss 미개선 시에 조기 종료
    patience = 12
    triggertimes = 0

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_f1_macro = 0
    best_f1_micro = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            if use_cutmix:
                ratio = np.random.rand(1)
                if ratio < cutmix_prob:
                    sample = cutmix.forward(inputs, labels)
                    outs = model(sample['image'])
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, sample['label_1'])*sample['lam'] + criterion(outs, sample['label_2']) * (1. - sample['lam'])
                else:
                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)
            else:
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4}|| lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)

                loss_value = 0

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
                
                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = MaskBaseDataset.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(inputs_np, labels, preds, n=16, shuffle=True)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            macro = f1_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), average="macro")  # f1_macro(preds, labels)
            micro = f1_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy(), average="micro")  # f1_micro(preds, labels) 
            
            if macro >= best_f1_macro:
                print(f"New best model for F1(macro)-Score : {macro:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best_macro.pth")
                best_f1_macro = macro
            if micro >= best_f1_micro:
                print(f"New best model for F1(micro)-Score : {micro:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best_micro.pth")
                best_f1_micro = micro
            
            best_val_loss = min(best_val_loss, val_loss)
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, F1(macro)-Score : {macro:4.2%}, F1(micro)-Score : {micro:4.2%}, loss: {val_loss:4.2} || "
                f"best F1(macro)-Score : {best_f1_macro:4.2%}, best F1(micro)-Score : {best_f1_micro:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1(macro)-score", macro, epoch)
            logger.add_scalar("Val/f1(micro)-score", micro, epoch)
            logger.add_figure("results", figure, epoch)

            # Early stopping
            if val_loss > best_val_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return model
            else:
                print('trigger times: 0')
                trigger_times = 0
            
            scheduler.step(val_loss)
            
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.3, help='ratio for validaton (default: 0.3)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--model', type=str, default='ResNet34', help='model type (default: ResNet34)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
