import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', isAug = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.class_map = {0: 0, 128: 1, 255: 2}
        self.isAug = isAug
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255
        if is_mask:
            # img_ndarray = np.uint8(img_ndarray)
            img_ndarray = np.uint8(img_ndarray).copy()
            # img_ndarray = img_ndarray / 128
            img_ndarray[img_ndarray == 128] = 1 # 视盘
            img_ndarray[img_ndarray == 255] = 2 # 视杯

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        # # 数据增强
        # img = np.array(img, dtype=np.uint8)
        # mask = np.array(mask, dtype=np.uint8)
        # if self.isAug:
        #     aug = Dataset_aug(image=img, mask=mask)
        #     img, mask = aug['image'], aug['mask']
        # 保存
        # img_save_test = Path('E:\\Dataset\\REFUGE\\512\\aug\\image\\')
        # mask_save_test = Path('E:\\Dataset\\REFUGE\\512\\aug\\mask\\')
        # cv2.imwrite(str(img_save_test.joinpath(name + '.png')), img)
        # cv2.imwrite(str(mask_save_test.joinpath(name + '.png')), mask)
        # 标签 检查
        # classes = np.unique(mask)
        # for each_class in classes:
        #     assert each_class in self.class_map.keys()

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'name': name
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
