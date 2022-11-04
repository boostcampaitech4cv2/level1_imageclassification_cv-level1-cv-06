from importlib import import_module
import argparse
import multiprocessing
import pandas as pd
import torch
import os
from dataset import TestDataset
import numpy as np
from tqdm import tqdm

def load_model(model_path, num_classes, device):
    model_module = getattr(import_module("model"), args.model)  # default: ViT16
    model = model_module(
        num_classes=num_classes
    ).to(device)


    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def infer(model_dir, data_dir, output_dir):
    model_list = sorted(os.listdir(model_dir))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 8  # 18


    all_mask_preds = []
    all_gender_preds = []
    all_age_preds = []
    for k, model_path in enumerate(model_list):
        k += 1


        model = load_model(os.path.join(model_dir, model_path, f'best_{k}.pth'), num_classes=num_classes, device=device)

        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print(f"Calculating for {k} fold model..")
        mask_preds = []
        gender_preds = []
        age_preds = []
        with torch.no_grad():
            for idx, images in tqdm(enumerate(loader)):
                images = images.to(device)
                all_preds = model(images).cpu()
                (mask_pred, gender_pred, age_pred) = torch.split(all_preds, [3, 2, 3], dim=1)           

                mask_preds.extend(mask_pred.numpy())
                gender_preds.extend(gender_pred.numpy())
                age_preds.extend(age_pred.numpy())

        
        all_mask_preds.append(np.array(mask_preds))
        all_gender_preds.append(np.array(gender_preds))
        all_age_preds.append(np.array(age_preds))


    
    all_mask_preds = np.array(all_mask_preds)
    all_gender_preds = np.array(all_gender_preds)
    all_age_preds = np.array(all_age_preds)


    mask_k = np.mean(all_mask_preds, axis=0)
    gender_k = np.mean(all_gender_preds, axis=0)
    age_k = np.mean(all_age_preds, axis=0)


    mask_k = mask_k.argmax(axis=-1)
    gender_k = gender_k.argmax(axis=-1)
    age_k = age_k.argmax(axis=-1)

    info['ans'] = mask_k * 6 + gender_k * 3 + age_k

    save_path = os.path.join(output_dir, 'output.csv')
    info.to_csv(save_path, index=False)
    print("Inference Done")
    






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ViT16', help='model type (default: ViT16)')

    # Container environment
    parser.add_argument('--data_dir', type=str, help= 'path where evaluation images exist')
    parser.add_argument('--model_dir', type=str, help='path where models for soft-voting exist')
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    model_dir = args.model_dir
    data_dir = args.data_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    infer(model_dir, data_dir, output_dir)