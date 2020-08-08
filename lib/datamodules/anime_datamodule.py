from argparse import ArgumentParser

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor

from pytorch_lightning import LightningDataModule

from .anime_dataset import AnimeDataset


class AnimeDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            num_workers: int):
        super(AnimeDataModule, self).__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.transform = Compose([
            ToTensor(),
            Normalize((1, 1)), #TODO Normalize
        ])

    def prepare_data(self):
        #TODO downloading data
        pass

    def setup(self, train_size, val_size, test_size): #here we create datasets
        filenames = [str(p) for p in Path(data_dir).glob('*.png')]

        train_val_test_dataset = AnimeDataset(
            filenames=filenames,
            transform=self.transform)
        self.train_val_dataset, self.test_dataset = random_split(
            dataset=train_val_test_dataset,
            lengths=(train_size + val_size, test_size))
        self.train_dataset, self.val_dataset = random_split(
            dataset=train_val_dataset,
            lengths=(train_size, val_size))

    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

        return test_dataloader

    @staticmethod
    def add_datamodule_specific_args(parent_parser, args_info):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        for arg_name in args_info:
            parser.add_argument(
                f'--{arg_name}',
                type=args_info[arg_name]['type'],
                default=args_info[arg_name]['default'])

        return parser

