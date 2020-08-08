from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class AnimeDataset(Dataset):
    def __init__(self, filenames, transform=[ToTensor()]):
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        img = Image.open(img_filename)

        if self.transform:
            img = self.transform(img)

        damaged_img = self.damaged(img)

        return (img, damaged_img)

    @staticmethod
    def damaged(x):
        return x * 0.5

