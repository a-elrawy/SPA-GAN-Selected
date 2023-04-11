import glob
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from utils import download_dataset

IMG_HEIGHT = 256
IMG_WIDTH = 256


def get_loader(dataset_name="facades", batch_size=1):
    """Builds and returns Dataloader for dataset_name.
    Args:
        dataset_name (str): Name of the dataset. One of 'facades', 'day2night', 'maps', 'cityscapes', 'horse2zebra',
        'apple2orange', 'summer2winter_yosemite', 'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo',
        'iphone2dslr_flower', 'ae_photos'

        batch_size (int): Batch size for training. Default: 1.
    """

    dataset_path = download_dataset(dataset_name)
    print(f"Dataset downloaded at {dataset_path}.")
    # Image preprocessing
    transforms_ = [
        T.Resize([286, 286], interpolation=T.InterpolationMode.NEAREST),
        T.RandomCrop(size=[IMG_HEIGHT, IMG_WIDTH]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Dataloader
    train_dataloader = DataLoader(
        ImageDataset(f"{dataset_path}", transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    test_dataloader = DataLoader(
        ImageDataset(f"{dataset_path}", transforms_=transforms_, mode="test"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return train_dataloader, test_dataloader


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = T.Compose(transforms_)

        self.files_A = sorted(glob.glob(f'{os.path.join(root, f"{mode}A")}/*.*'))
        self.files_B = sorted(glob.glob(f'{os.path.join(root, f"{mode}B")}/*.*'))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])

        if np.random.random() < 0.5:
            image_A = Image.fromarray(np.array(image_A)[:, ::-1, :], "RGB")
            image_B = Image.fromarray(np.array(image_B)[:, ::-1, :], "RGB")

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return item_A, item_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
