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
        lr=3e-4,
        n_channels=3,
        n_classes=2)

    datamodule = AnimeDataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers)
    datamodule.setup(
        test_ratio=args.test_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio)

    trainer = Trainer.from_argparse_args(
        args=args,
        auto_lr_find=args.auto_lr_find,
        logger=False)

    print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    print(len(datamodule.train_dataloader()))
    print(len(datamodule.val_dataloader()))

    #lr_finder = trainer.lr_find(model)
    #model.learning_rate = lr_finder.suggestion()

    trainer.fit(
        model=model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader())
    print('AAAAAABBBBBBBBBBBBBBBBBBBBBBBBAAAAAAAAAAAAAAAA')
    trainer.test(
        test_dataloaders=datamodule.test_dataloader(),
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

