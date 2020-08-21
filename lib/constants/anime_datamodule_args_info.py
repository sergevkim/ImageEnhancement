from pathlib import Path


ANIME_DATAMODULE_ARGS_INFO = {
    'batch_size': {
        'default': 10,
        'type': int,
    },
    'data_dir': {
        'default': f'{Path.cwd()}/data/images',
        'type': str,
    },
    'num_workers': {
        'default': 4,
        'type': int,
    },
    'test_ratio': {
        'default': 0.1,
        'type': int,
    },
    'train_ratio': {
        'default': 0.7,
        'type': int,
    },
    'val_ratio': {
        'default': 0.2,
        'type': int,
    },
}

