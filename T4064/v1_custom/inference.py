import argparse
import multiprocessing
import os
import pandas as pd
import torch
import timm
import albumentations as A
# import model


from albumentations import *
from albumentations.pytorch import ToTensorV2
from importlib import import_module
from torch.utils.data import DataLoader
from dataset import CustomTestDataset


def load_model(model_dir, checkpoint_name, model_name, n_classes, device):
    model_path = os.path.join(model_dir, checkpoint_name, 'best.pth')
    model = timm.create_model(model_name=model_name, 
                              num_classes=n_classes, 
                              pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


@torch.no_grad()
def inference(data_dir, model_dir, checkpoint_name, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.is_custom_model == "No":
        model = load_model(model_dir, checkpoint_name, args.model_name, args.n_classes, device).to(device)
    else: ## if custom model, how import...?
        print('custom model')
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    ## --- transforms
    test_transforms = A.Compose([A.Resize(height=224, width=224),
                                 A.Normalize(mean=(0.56019358,0.52410121,0.501457),
                                 std=(0.23318603,0.24300033,0.24567522)),
                                 ToTensorV2()])

    dataset = CustomTestDataset(img_paths, test_transforms)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'{args.checkpoint_name}.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--is_custom_model', type=str, default="No", help='if you make your custom model check please')
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3_pruned')
    parser.add_argument('--n_classes', type=int, default=18, help='number_of_class')
    parser.add_argument('--checkpoint_name', type=str, default='good_night')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', './T4064/dataset/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './T4064/checkpoints'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './T4064/submission'))
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    checkpoint_name = args.checkpoint_name

    for arg in vars(args):
        print(arg, getattr(args, arg))

    os.makedirs(output_dir, exist_ok=True)
    # inference(data_dir, model_dir, checkpoint_name, output_dir, args):
    inference(data_dir, model_dir, checkpoint_name, output_dir, args)
