import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from stat import ST_GID

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

from sklearn.metrics import f1_score


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


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

    save_dir = increment_path(os.path.join(model_dir, args.name))

    ## -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ## -- dataset
    # dataset_module = dataset.MaskBaseDataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    # dataset = dataset.MaskBaseDataset(data_dir=data_dir)
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    ## -- augmentation
    # transform_module = dataset.BaseAugmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    ## -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    ## -- model
    # model_module = model.BaseModel
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    # model = model.BaseModel(num_classes=num_classes)
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model) 

    ## -- loss & metric
    # criterion = nn.CrossEntropyLoss(**kwargs)
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    # opt_module = torch.optim.SGD
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    # optimizer = SGD(model.parameters(), lr=lr, weight_decay=5e-4)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )


    ## -- scheduler
    if args.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.5)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
         raise ValueError   
    
    ## -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    # torch.save(args.note, f"{save_dir}/note.txt")
    with open(os.path.join(save_dir, 'note.json'), 'w', encoding='utf-8') as f:
        json.dump(args.note, f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0
    
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        for idx, train_batch in enumerate(train_loader):
            
            # NOT MULTI-LABEL
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model.forward(inputs)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            preds = torch.argmax(outs, dim=-1)

            
            
            
            # # MULTI-LABEL
            # inputs, (mask_labels, gender_labels, age_labels) = train_batch
            # inputs = inputs.to(device)
            # mask_labels = mask_labels.to(device)
            # gender_labels = gender_labels.to(device)
            # age_labels = age_labels.to(device)
            
            # optimizer.zero_grad()
            
            # outs = model.forward(inputs)
            # mask_outs, gender_outs, age_outs = torch.split(outs, [3,2,3], dim=1)
            # mask_loss = criterion(mask_outs, mask_labels)
            # gender_loss = criterion(gender_outs, gender_labels)
            # age_loss = criterion(age_outs, age_labels)
            # loss = mask_loss + gender_loss + 1.2*age_loss       # if model sucks as guessing 'age'
            
            # loss.backward()
            # optimizer.step()
            
            # loss_value += loss.item()
            # preds = 
            
            
            
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        # scheduler.step()      # -->> for stepLR
        

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            
            # for f1 score
            epoch_preds = []
            epoch_labels = []
            
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

                # for f1 score
                epoch_preds += preds.detach().cpu().numpy().tolist()
                epoch_labels += labels.detach().cpu().numpy().tolist()
                
                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                    

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 = f1_score(epoch_labels, epoch_preds, average='micro')
            
            scheduler.step(val_loss)
            
            best_val_loss = min(best_val_loss, val_loss)            
            if val_acc > best_val_acc:
                # print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                # torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            if val_f1 > best_val_f1:
                print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
                
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 1')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='focal', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='scheduler: StepLR, CosineAnnealingLR, ReduceLROnPlateau...')
    parser.add_argument('--note', type=str)


    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)
    # print('))))' * 50)
    # print(vars(args))
    # print('))))' * 50)

    # from T4064
    for arg in vars(args):
        print(arg, "-->>",getattr(args, arg))

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)




# loss(criterion): focal
# optimizer: AdamW {default: SGD}
# scheduler: ReduceLROnPlateau(hard coded)



# python train.py --model MyResnet34 --epochs 50 --augmentation BaseAugmentation_size --criterion focal --optimizer AdamW --name 'Resnet34_02' --note 'model:MyResnet34, epoch:50, augmentation:Soft, loss:focal, optimizer:AdamW, scheduler:reduceplateau'
# python train.py --model swin_base_patch4_window7_224 --epochs 50 --augmentation BaseAugmentation_size --criterion focal --optimizer AdamW --name 'swin_base_patch4_window7_224' --note 'model:swin_base_patch4_window7_224, epoch:50, augmentation:Soft, loss:focal, optimizer:AdamW, scheduler:reduceplateau'



