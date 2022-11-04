import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import pandas as pd
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskMultiLabelDataset
from loss import create_criterion, LabelSmoothingLoss

from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_dataloader(df, transform, batch_size, shuffle):
    dataset = MaskMultiLabelDataset(df=df, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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


def fold_train(whole_df, model_dir, args):
    train_transform = A.Compose([
        A.CenterCrop(384, 384),
        A.RandomCrop(320, 320, p=0.5),
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
                A.RandomBrightnessContrast(),
                A.CLAHE(),
                A.ToGray(),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(0.002),
            A.MedianBlur(blur_limit=3),
            A.Blur(blur_limit=3)
        ], p=0.3),
        A.RandomRotate90(p=0.1),
        A.CoarseDropout(max_holes=10, min_holes = 7, p=0.2),
        A.Rotate(limit=20, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.CenterCrop(384, 384),
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


    # -- k-fold
    k = 1
    id_df = whole_df.drop_duplicates(subset = 'ID')
    kfold = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)

    for train_id_idx, val_id_idx in kfold.split(id_df['ID'], id_df['label']):
        print(f'================={k} fold training=================')
        train_id = id_df.iloc[train_id_idx]['ID']
        val_id = id_df.iloc[val_id_idx]['ID']
        train_df = whole_df[whole_df['ID'].isin(train_id)]
        val_df = whole_df[whole_df['ID'].isin(val_id)]
        train_loader = get_dataloader(df=train_df, transform=train_transform, batch_size=args.batch_size, shuffle=True)
        val_loader = get_dataloader(df=val_df, transform=val_transform, batch_size=args.batch_size, shuffle=False)
        train(k, train_loader, val_loader, model_dir, args)
        k += 1


def encode_labels(mask, gender, age):
    #012 01 012
    mask = mask.tolist()
    gender = gender.tolist()
    age = age.tolist()
    mask = list(map(str, mask))
    gender = list(map(str, gender))
    age = list(map(str, age))
    label = list(map(lambda x,y,z : int(str(x)+str(y)+str(z)), mask,gender,age))

    return torch.tensor(label)



def train(k, train_loader, val_loader, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 8
    


    # -- model
    
    model_module = getattr(import_module("model"), args.model)  # default: Effb4
    model = model_module(
        num_classes=num_classes
    ).to(device)
    
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: focal
    age_criterion = LabelSmoothingLoss()
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: AdamW
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    #RLROP metric: val set f1 score
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, mode='max', patience=args.lr_reduce_patience)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # early_stopping : epoch 연속으로 loss 미개선 시에 조기 종료
    stop_patience = args.early_stop_patience
    trigger_times = 0

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        
        epoch_preds = []
        epoch_labels = []
        
        for idx, train_batch in enumerate(train_loader):
            inputs, (mask_labels, gender_labels, age_labels) = train_batch
            inputs = inputs.to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)
            labels = encode_labels(mask_labels, gender_labels, age_labels)


            optimizer.zero_grad()

            outs = model(inputs)
            (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)
            mask_preds = torch.argmax(mask_outs, dim=-1)
            gender_preds = torch.argmax(gender_outs, dim=-1)
            age_preds = torch.argmax(age_outs, dim=-1)
            preds = encode_labels(mask_preds, gender_preds, age_preds)

            mask_loss = criterion(mask_outs, mask_labels)
            gender_loss = criterion(gender_outs, gender_labels)
            age_loss = age_criterion(age_outs, age_labels)


            loss = mask_loss + gender_loss + (1.5 * age_loss)
            loss.backward()
            
            optimizer.step()

            loss_value += loss.item()
            matches += (preds==labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                try: current_lr = get_lr(optimizer)
                except:
                    current_lr = 'undefined'
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr:0.8}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

#         scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, (mask_labels, gender_labels, age_labels) = val_batch
                inputs = inputs.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)
                labels = encode_labels(mask_labels, gender_labels, age_labels)

                outs = model(inputs)
                (mask_outs, gender_outs, age_outs) = torch.split(outs, [3, 2, 3], dim=1)
                mask_preds = torch.argmax(mask_outs, dim=-1)
                gender_preds = torch.argmax(gender_outs, dim=-1)
                age_preds = torch.argmax(age_outs, dim=-1)
                preds = encode_labels(mask_preds, gender_preds, age_preds)

                mask_loss = criterion(mask_outs, mask_labels)
                gender_loss = criterion(gender_outs, gender_labels)
                age_loss = age_criterion(age_outs, age_labels)
                
                loss_item = (mask_loss+gender_loss+(1.5*age_loss)).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                
                epoch_preds += preds.detach().cpu().numpy().tolist()
                epoch_labels += labels.detach().cpu().numpy().tolist()



            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / (len(val_loader) * args.batch_size)
            val_f1 = f1_score(epoch_labels, epoch_preds, average='macro')
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                trigger_times = 0
                torch.save(model.state_dict(), f"{save_dir}/best_{k}.pth")
                best_val_f1 = val_f1

            else:
                trigger_times += 1

            torch.save(model.state_dict(), f"{save_dir}/last_{k}.pth")
            print('trigger times', trigger_times)
            if trigger_times >= stop_patience:
                print('early stopped' + " " * 30)
                break
            scheduler.step(val_f1)
            
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"[Val] f1 : {val_f1:2.4} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskMultiLabelDataset', help='dataset augmentation type (default: MaskMultiLabelDataset)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='Effb4', help='model type (default: Effb4)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: focal)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--n_fold', type=int, default = 5)
    parser.add_argument('--lr_reduce_patience', type=int, default = 7)
    parser.add_argument('--early_stop_patience', type=int, default = 10)

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--csv_path', type=str, default = './custom_train_adjusted.csv')

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    whole_df = pd.read_csv(args.csv_path, index_col=[0])
    fold_train(whole_df, model_dir, args)
