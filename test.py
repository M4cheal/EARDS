import cv2
from PIL import Image
import argparse
import logging
import sys
from pathlib import Path
from torchvision import transforms
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
# from unet import UNet
from EARDS_model import model


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def predict_img(net, image, mask, out_threshold=0.5):
    net.eval()
    image = image.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device, dtype=torch.long)
    mask = F.one_hot(mask, net.n_classes).permute(0, 3, 1, 2).float()

    dice_score = 0
    # dice_score_od = 0
    # dice_score_oc = 0
    with torch.no_grad():
        output = net(image)

        if net.n_classes == 1:
            mask_pred = (F.sigmoid(output) > 0.5).float()
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask, reduce_batch_first=False)
        else:
            mask_pred = F.one_hot(output.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask[:, 1:, ...], reduce_batch_first=False)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy(), dice_score
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy(), dice_score
netn_channels = 3

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # 图像地址
    in_img_files = Path('///')
    # 标签地址
    in_mask_files = Path('///')
    # 输出地址
    out_files = '///'
    # 模型地址
    dir_checkpoint = Path('checkpoints/EARDS/checkpoint_epoch30.pth')
    net = model.EARDS(model='b0', out_channels=3, freeze_backbone=True, pretrained=False, device='cuda', num_gpu=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {dir_checkpoint}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(dir_checkpoint, map_location=device))

    logging.info('Model loaded!')

    dataset = BasicDataset(in_img_files, in_mask_files, args.scale)

    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=True, **loader_args)
    dice_score_sum = 0
    for batch in test_loader:
        image = batch["image"]
        mask = batch["mask"]
        name = str(batch["name"][0])
        logging.info(f'\nPredicting image {batch["name"]} ...')
        pre_mask, dice_score = predict_img(net, image, mask)
        logging.info(f'{dice_score}')

        out6 = torch.max(torch.tensor(pre_mask), 0)
        index6 = out6.indices
        n26 = index6.numpy()
        target = n26
        target = target.astype(np.uint8)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)
        new_label = np.zeros(target.shape, dtype=np.int64)
        new_label[target == 0] = 0
        new_label[target == 1] = 128
        new_label[target == 2] = 255
        target = new_label
        cv2.imwrite(out_files + name + '.png', target)

        dice_score_sum += dice_score


    dice_score_ave = dice_score_sum / len(test_loader)
    print(f'{dice_score_ave} = {dice_score_sum} / {len(test_loader)}')