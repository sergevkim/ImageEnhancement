from argparse import ArgumentParser

import torch
from torch.nn import Module, Sequential
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    MSELoss,
    ReLU)
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule


class Block(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int):
        super().__init__()
        self.block = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class BlockDown(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int):
        super().__init__()
        self.block_down = Sequential(
            Block(in_channels=in_channels, out_channels=out_channels), #TODO mid_channels == in_channels // 2 == out_channels * 2
            Block(in_channels=out_channels, out_channels=out_channels),
            MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.block_down(x)


class BlockUp(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int):
        super().__init__()
        self.block_down = Sequential(
            Block(in_channels=in_channels, out_channels=out_channels),
            Block(in_channels=out_channels, out_channels=out_channels),
            ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2))

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)

        return self.block_down(x)


class UNetModel(LightningModule):
    def __init__(
            self,
            n_channels: int,
            n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        block_down_1 = BlockDown(n_channels, 10),
        block_down_2 = BlockDown(10, 20),
        block_down_3 = BlockDown(20, 30)
        block_down_4 = BlockDown(20, 30)

        block_up_1 = BlockUp(40, 30)
        block_up_2 = BlockUp(30, 20)
        block_up_3 = BlockUp(20, 10)
        block_up_4 = BlockUp(10, n_classes)

    def forward(self, x_0):
        x_1 = self.block_down_1(x_0)
        x_2 = self.block_down_2(x_1)
        x_3 = self.block_down_3(x_2)
        x_4 = self.block_down_4(x_3)

        x = self.block_up_1(x_4, x_3)
        x = self.block_up_2(x, x_2)
        x = self.block_up_3(x, x_1)
        x = self.block_up_4(x, x_0)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = MSELoss(y_hat, y)

        tensorboard_logs = {'train_loss': loss}
        output = {
            'loss': loss,
            'log': tensorboard_logs,
        }

        return output

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx): #TODO
        x, y = batch
        y_hat = self.forward(x)

        loss = MSELoss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)

        output = {
            'val_loss': loss,
        }

        return output

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        self.optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate)
        self.scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=10)

        return [self.optimizer], [self.scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser, args_info):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        for arg_name in args_info:
            parser.add_argument(
                f'--{arg_name}',
                type=args_info[arg_name]['type'],
                default=args_info[arg_name]['default'])

        return parser

