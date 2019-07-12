from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import numpy as np
import random


class SatelliteDataset(Dataset):
    def __init__(self, root, hr_patch_size, scale_factor=2):
        self.root = root
        self.fnames = os.listdir(self.root)
        self.hr_tfms = transforms.Compose([
                        transforms.RandomChoice([transforms.Resize(hr_patch_size),
                                                 transforms.RandomCrop(hr_patch_size)]),

                        ])
        lr_patch_size = (hr_patch_size[0] // scale_factor, hr_patch_size[1] // scale_factor)

        # Scale factor to resize image to a small size
        # Such that when it is rescaled, the image is pixelated
        # https://stackoverflow.com/questions/47143332/how-to-pixelate-a-square-image-to-256-big-pixels-with-python
        pixelation_factor = 4 * scale_factor
        pixelation_size = (hr_patch_size[0] // pixelation_factor, hr_patch_size[1] // pixelation_factor)
        self.lr_tfms = transforms.Compose([
                        transforms.RandomChoice([transforms.Resize(pixelation_size, interpolation=Image.NEAREST),
                                                 transforms.Resize(pixelation_size, interpolation=Image.BILINEAR)]),
                        transforms.Resize(lr_patch_size, interpolation=Image.NEAREST)
                        ])

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def load_image(path):
        img = Image.open(path)
        # Convert to 3-Channel
        img = img.convert('RGB')
        return img

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.fnames[index])
        image = SatelliteDataset.load_image(img_path)
        hres = self.hr_tfms(image)
        lres = self.lr_tfms(hres)

        # Perform random horizontal and vertical flip
        if random.random() > 0.5:
            hres.transpose(Image.FLIP_LEFT_RIGHT)
            lres.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            hres.transpose(Image.FLIP_TOP_BOTTOM)
            lres.transpose(Image.FLIP_TOP_BOTTOM)

        return self.to_tensor(lres), self.to_tensor(hres)


class SatelliteValDataset(Dataset):
    def __init__(self, root, hr_patch_size, scale_factor=2):
        self.root = root
        self.fnames = os.listdir(self.root)
        self.hr_tfms = transforms.Compose([transforms.Resize(hr_patch_size)])
        lr_patch_size = (hr_patch_size[0] // scale_factor, hr_patch_size[1] // scale_factor)

        # Scale factor to resize image to a small size
        # Such that when it is rescaled, the image is pixelated
        # https://stackoverflow.com/questions/47143332/how-to-pixelate-a-square-image-to-256-big-pixels-with-python
        pixelation_factor = 4 * scale_factor
        pixelation_size = (hr_patch_size[0] // pixelation_factor, hr_patch_size[1] // pixelation_factor)
        self.lr_tfms = transforms.Compose([
                        transforms.RandomChoice([transforms.Resize(pixelation_size, interpolation=Image.NEAREST),
                                                 transforms.Resize(pixelation_size, interpolation=Image.BILINEAR)]),
                        transforms.Resize(lr_patch_size, interpolation=Image.NEAREST)
                        ])

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def load_image(path):
        img = Image.open(path)
        # Convert to 3-Channel
        img = img.convert('RGB')
        return img

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.fnames[index])
        image = SatelliteDataset.load_image(img_path)
        hres = self.hr_tfms(image)
        lres = self.lr_tfms(hres)
        return self.to_tensor(lres), self.to_tensor(hres)


if __name__ == '__main__':
    # Test dataloader
    ROOT = 'data/hr'
    sat_data = SatelliteDataset(ROOT, (512, 512), 2)
    lr, hr = sat_data[109]
    to_pil = transforms.ToPILImage()
    cv2.imshow('HIGH-RES', np.array(to_pil(hr)))
    cv2.waitKey(5000)
    cv2.imshow('LOW-RES', np.array(to_pil(lr)))
    cv2.waitKey(5000)
