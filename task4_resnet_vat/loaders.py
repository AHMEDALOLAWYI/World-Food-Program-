from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    return img


class SatClassificationDataset(Dataset):
    def __init__(self, img_dir, split_csv, positive_class, is_val,
                 smoothing_factor=None, tfms=None):
        df = pd.read_csv(split_csv)
        df['label'] = df['fname'].apply(lambda x: int(x.startswith(positive_class)))
        self.subset = df[df['is_val_set'] == is_val]
        self.img_names = df['fname'].values
        self.labels = df['label'].values
        self.smoothing_factor = smoothing_factor
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image = load_image(os.path.join(self.img_dir, self.img_names[idx]))
        if self.tfms is not None:
            image = self.tfms(image)
        return image, self.labels[idx]


class UnlabeledDataset(Dataset):
    def __init__(self, root, size):
        self.root = root
        self.fnames = os.listdir(self.root)
        self.tfms = transforms.Compose([
            transforms.RandomChoice([transforms.Resize(size),   # High quality full image
                                     transforms.RandomCrop(size), # High quality random patch
                                     transforms.Compose(
                                         [transforms.Resize(size // 2, interpolation=Image.BILINEAR),
                                          transforms.Resize(size, interpolation=Image.NEAREST)]), # Low quality full image
                                     transforms.Compose(
                                         [transforms.RandomCrop(size // 2),
                                          transforms.Resize(size, interpolation=Image.NEAREST)])  # Low quality random patch
                                     ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.fnames[index])
        image = load_image(img_path)
        image = self.tfms(image)
        return image, -1.0
