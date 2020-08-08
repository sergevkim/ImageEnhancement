from argparse import ArgumentParser
#from pathlib import Path
#import sys
#sys.path.append(f'{Path.cwd()}/.')

from pytorch_lightning import Trainer

from lib.constants import ANIME_DATAMODULE_ARGS_INFO, UNET_MODEL_ARGS_INFO
from lib.datamodules import AnimeDataModule
from lib.models import UNetModel


def main(args):
    dargs = vars(args)
    for key in dargs:
        print(key, dargs[key])

    model = UNetModel(
        learning_rate=args.learning_rate,
        n_channels=3,
        n_classes=2)

    datamodule = AnimeDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    datamodule.setup(
        data_dir=args.data_dir,
        test_size=args.test_size,
        train_size=args.train_size,
        val_size=args.val_size)

    trainer = Trainer.from_argparse_args(
        args=args,
        auto_lr_find=True,
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

