from argparse import ArgumentParser
#from pathlib import Path
#import sys
#sys.path.append(f'{Path.cwd()}/.')

from pytorch_lightning import Trainer

from datamodules import AnimeDataModule
from models import UNetModel
from constants import ANIME_DATAMODULE_ARGS_INFO, UNET_MODEL_ARGS_INFO


def main(args):
    model = UNetModel(
        hparams=args,
        n_channels=3,
        n_classes=2)

    datamodule = AnimeDataModule()
    datamodule.setup() #TODO setup or init

    trainer = Trainer.from_argparse_args(
        args=args,
        logger=False)
    trainer.fit(
        model=model,
        train_dataloader=datamodule.train_dataloader,
        val_dataloaders=datamodule.val_dataloader)
    trainer.test(
        test_dataloaders=datamodule.test_dataloader,
        ckpt_path='best')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parent_parser=parser)
    parser = UNetModel.add_model_specific_args(
        parent_parser=parser,
        args_info=UNET_MODEL_ARGS_INFO)
    parser = AnimeDataModule.add_datamodule_specific_args(
        parent_parser=parser,
        args_info=ANIME_DATAMODULE_ARGS_INFO)
    args = parser.parse_args()

    main(args)

